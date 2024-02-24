#include <fstream>
#include <iostream>
#include <vector>

#include "nn.h"

constexpr size_t imageSize = 784;

// Hyperparameters
constexpr float initialLearningRate = 0.0001f;
constexpr int lrDecayEpoch = 20;
constexpr float lrDecayRate = 0.5f;
constexpr size_t batchSize = 25;

std::vector<Tensor<float>> loadImages(std::string path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Cannot load image file {}", path));
  }

  file.seekg(16, std::ios::beg); // Skip header.
  std::vector<Tensor<float>> images;

  unsigned char buffer[imageSize];
  while (file.read(reinterpret_cast<char *>(buffer), imageSize)) {
    float image[imageSize];
    for (size_t i = 0; i < imageSize; ++i) {
      image[i] = buffer[i] / 255.0f;
    }
    images.emplace_back(
        Tensor<float>::vector(std::begin(image), std::end(image)));
  }
  return images;
}

std::vector<Tensor<float>> loadLabels(std::string path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Cannot load labels file {}", path));
  }

  file.seekg(8, std::ios::beg); // Skip header.

  std::vector<Tensor<float>> labels;

  char label;

  while (file.read(&label, 1)) {
    float labelOneHot[10] = {};
    labelOneHot[static_cast<size_t>(label)] = 1.0f;
    labels.emplace_back(
        Tensor<float>::vector(std::begin(labelOneHot), std::end(labelOneHot)));
  }
  return labels;
}

int main() {
  auto trainImages = loadImages("../data/mnist/train-images-idx3-ubyte");
  auto validationImages = loadImages("../data/mnist/t10k-images-idx3-ubyte");

  auto trainLabels = loadLabels("../data/mnist/train-labels-idx1-ubyte");
  auto validationLabels = loadLabels("../data/mnist/t10k-labels-idx1-ubyte");

  const size_t numTrainImages = trainImages.size();
  const size_t numValidationImages = validationImages.size();
  const size_t numBatches = numTrainImages / batchSize;

  LinearLayer layer0(imageSize, 16);
  ReLU layer0Activation;
  LinearLayer layer1(16, 16);
  ReLU layer0Activation;
  LinearLayer layer2(16, 16);
  ReLU layer2Activation;
  LinearLayer layer3(16, 10);

  float learningRate = initialLearningRate;
  for (int epoch = 0; epoch < 1000; epoch++) {
    for (size_t batch = 0; batch < numBatches; batch++) {
      // Forward pass.
      auto z0 = layer0.forward(datasetValues);
      auto a0 = layer0Activation.forward(z0);
      auto z1 = layer1.forward(a0);
      auto a1 = layer1Activation.forward(z1);
      auto result = layer2.forward(a1);

      // Accuracy and loss calculation.
      int correct = 0;
      for (size_t i = 0; i < datasetValues.getShape()[0]; i++) {
        if (std::signbit(result[i].item()) ==
            std::signbit(datasetLabels[i].item())) {
          correct++;
        }
      }
      float accuracy =
          static_cast<float>(correct) / datasetValues.getShape()[0];
      fmt::println("Accuracy: {}", accuracy);

      auto flatResult = result.reshape({100});
      auto loss = (flatResult - datasetLabels).reshape({100, 1});

      auto lossSum = Tensor<float>::ones({1, loss.shape[0]}).matmul(loss);
      fmt::println("Loss: {}", lossSum[0].item());

      // Backwards pass.
      layer0.zeroGrad();
      layer1.zeroGrad();
      layer2.zeroGrad();

      /// Layer 2
      auto outGrad = layer2.backward(a1, loss);

      /// Layer 1
      outGrad = layer1Activation.backward(z1, outGrad);
      outGrad = layer1.backward(a0, outGrad);

      // Layer 0
      outGrad = layer0Activation.backward(z0, outGrad);
      outGrad = layer0.backward(datasetValues, outGrad);

      // Gradient descent.
      layer2.weight = layer2.weight - layer2.weightGrad * learningRate;
      layer2.bias = layer2.bias - layer2.biasGrad * learningRate;

      layer1.weight = layer1.weight - layer1.weightGrad * learningRate;
      layer1.bias = layer1.bias - layer1.biasGrad * learningRate;

      layer0.weight = layer0.weight - layer0.weightGrad * learningRate;
      layer0.bias = layer0.bias - layer0.biasGrad * learningRate;

      if (epoch % lrDecayEpoch == 0 && epoch != 0) {
        learningRate *= lrDecayRate;
      }
    }
  }

  return 0;
}
