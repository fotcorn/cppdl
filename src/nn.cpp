#include "data/nn.h"
#include "data/dataset.h"

#include "nn.h"

int main() {
  LinearLayer layer0(2, 16);
  ReLU layer0Activation;
  LinearLayer layer1(16, 16);
  ReLU layer1Activation;
  LinearLayer layer2(16, 1);

  float learningRate = 0.0001f;

  for (int epoch = 0; epoch < 1000; epoch++) {
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
    float accuracy = static_cast<float>(correct) / datasetValues.getShape()[0];
    fmt::println("Accuracy: {}", accuracy);

    auto flatResult = result.reshape({100});
    auto loss = (flatResult - datasetLabels).reshape({100, 1});

    // Backwards pass.
    auto lossSum = Tensor<float>::ones({1, loss.shape[0]}).matmul(loss);
    fmt::println("Loss: {}", lossSum[0].item());
    layer2.biasGrad = layer2.biasGrad + lossSum;
    layer2.weightGrad = layer2.weightGrad + loss.transpose().matmul(a1);

    auto sp = layer1Activation.backward(z1);
    auto delta = loss.matmul(layer2.weight) * sp;
    auto deltaSum = Tensor<float>::ones({1, delta.shape[0]}).matmul(delta);
    layer1.biasGrad = layer1.biasGrad + deltaSum;
    layer1.weightGrad = layer1.weightGrad + delta.transpose().matmul(a0);

    sp = layer0Activation.backward(z0);
    delta = delta.matmul(layer1.weight) * sp;
    deltaSum = Tensor<float>::ones({1, delta.shape[0]}).matmul(delta);
    layer0.biasGrad = layer0.biasGrad + deltaSum;
    layer0.weightGrad =
        layer0.weightGrad + delta.transpose().matmul(datasetValues);

    // Gradient descent.

    layer2.weight = layer2.weight - layer2.weightGrad * learningRate;
    layer2.bias = layer2.bias - layer2.biasGrad * learningRate;

    layer1.weight = layer1.weight - layer1.weightGrad * learningRate;
    layer1.bias = layer1.bias - layer1.biasGrad * learningRate;

    layer0.weight = layer0.weight - layer0.weightGrad * learningRate;
    layer0.bias = layer0.bias - layer0.biasGrad * learningRate;

    if (epoch % 20 == 0 && epoch != 0) {
      learningRate *= 0.5f;
    }
  }
  return 0;
}
