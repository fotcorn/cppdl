#include <fstream>
#include <iostream>
#include <vector>

#include "cppdl/mnist_utils.h"
#include "cppdl/nn.h"
#include "cppdl/serialization.h"

constexpr size_t imageSize = 784;

// Hyperparameters
constexpr float initialLearningRate = 0.001f;
constexpr int lrDecayEpoch = 30;
constexpr float lrDecayRate = 0.5f;
constexpr size_t batchSize = 200;

int main() {
  auto rawTrainImages = loadImages("../data/mnist/train-images-idx3-ubyte");
  auto rawTrainLabels = loadLabels("../data/mnist/train-labels-idx1-ubyte");

  std::vector<Tensor<float>> trainImages, trainLabels;
  for (size_t i = 0; i < rawTrainImages.size(); i += batchSize) {
    auto batchEnd = std::min(rawTrainImages.size(), i + batchSize);
    std::vector<Tensor<float>> batchImages(rawTrainImages.begin() + i,
                                           rawTrainImages.begin() + batchEnd);
    trainImages.push_back(
        Tensor<float>::stack(batchImages.begin(), batchImages.end()));

    std::vector<Tensor<float>> batchLabels(rawTrainLabels.begin() + i,
                                           rawTrainLabels.begin() + batchEnd);
    trainLabels.push_back(
        Tensor<float>::stack(batchLabels.begin(), batchLabels.end()));
  }

  auto rawValidationImages = loadImages("../data/mnist/t10k-images-idx3-ubyte");
  auto rawValidationLabels = loadLabels("../data/mnist/t10k-labels-idx1-ubyte");
  auto validationImages = Tensor<float>::stack(rawValidationImages.begin(),
                                               rawValidationImages.end());
  auto validationLabels = Tensor<float>::stack(rawValidationLabels.begin(),
                                               rawValidationLabels.end());

  const size_t numTrainImages = rawTrainImages.size();
  const size_t numBatches = numTrainImages / batchSize;

  LinearLayer layer0(imageSize, 16);
  ReLU layer0Activation;
  LinearLayer layer1(16, 16);
  ReLU layer1Activation;
  LinearLayer layer2(16, 10);

  float learningRate = initialLearningRate;
  for (int epoch = 1; epoch < 10000000; epoch++) {
    int trainCorrect = 0;
    float trainLoss = 0.0f;

    for (size_t batch = 0; batch < numBatches; batch++) {
      // Forward pass.
      auto z0 = layer0.forward(trainImages[batch]);
      auto a0 = layer0Activation.forward(z0);
      auto z1 = layer1.forward(a0);
      auto a1 = layer1Activation.forward(z1);
      auto z2 = layer2.forward(a1);
      auto result = z2.softmax(1);

      // Loss calculation.
      auto loss = result - trainLabels[batch];
      trainLoss += result.meanSquareError(trainLabels[batch]);

      // Accuracy calculation.
      auto predictedLabels = result.argmax(1);
      auto trueLabels = trainLabels[batch].argmax(1);

      for (size_t i = 0; i < predictedLabels.size; i++) {
        if (predictedLabels.data[i] == trueLabels.data[i]) {
          trainCorrect++;
        }
      }

      // Backwards pass.
      layer0.zeroGrad();
      layer1.zeroGrad();
      layer2.zeroGrad();

      /// Layer 2
      // outGrad = layer2Activation.backward(z2, outGrad);
      auto outGrad = layer2.backward(a1, loss);

      /// Layer 1
      outGrad = layer1Activation.backward(z1, outGrad);
      outGrad = layer1.backward(a0, outGrad);

      // Layer 0
      outGrad = layer0Activation.backward(z0, outGrad);
      outGrad = layer0.backward(trainImages[batch], outGrad);

      // Gradient descent.
      layer2.weight = layer2.weight - layer2.weightGrad * learningRate;
      layer2.bias = layer2.bias - layer2.biasGrad * learningRate;

      layer1.weight = layer1.weight - layer1.weightGrad * learningRate;
      layer1.bias = layer1.bias - layer1.biasGrad * learningRate;

      layer0.weight = layer0.weight - layer0.weightGrad * learningRate;
      layer0.bias = layer0.bias - layer0.biasGrad * learningRate;
    }

    if (epoch % lrDecayEpoch == 0) {
      learningRate *= lrDecayRate;
      fmt::println("New learning rate: {}", learningRate);
    }

    float accuracy = static_cast<float>(trainCorrect) / numTrainImages;

    fmt::println("Epoch: {}, Train Loss: {}, Train Accuracy: {}", epoch,
                 trainLoss / numBatches, accuracy);

    if (epoch % 10 == 0) {
      auto z0 = layer0.forward(validationImages);
      auto a0 = layer0Activation.forward(z0);
      auto z1 = layer1.forward(a0);
      auto a1 = layer1Activation.forward(z1);
      auto z2 = layer2.forward(a1);
      auto result = z2.softmax(1);

      float valLoss = result.meanSquareError(validationLabels);

      auto predictedLabels = result.argmax(1);
      auto trueLabels = validationLabels.argmax(1);

      size_t valCorrect = 0;
      for (size_t i = 0; i < predictedLabels.size; i++) {
        if (predictedLabels.data[i] == trueLabels.data[i]) {
          valCorrect++;
        }
      }

      float valAccuracy =
          static_cast<float>(valCorrect) / rawValidationImages.size();
      fmt::println("Validation - Epoch: {}, Loss: {}, Accuracy: {}", epoch,
                   valLoss, valAccuracy);

      std::ofstream file;
      std::string filename =
          fmt::format("mnist-{}-{:.2f}.cppdl", epoch, valAccuracy * 100);
      file.open(filename, std::ios::binary);
      if (!file.is_open()) {
        fmt::println("Error opening file: {}", filename);
        return -1;
      }

      writeFileHeader(file);

      serializeTensor(layer0.weight, file);
      serializeTensor(layer0.bias, file);
      serializeTensor(layer1.weight, file);
      serializeTensor(layer1.bias, file);
      serializeTensor(layer2.weight, file);
      serializeTensor(layer2.bias, file);

      file.close();
    }
  }

  return 0;
}
