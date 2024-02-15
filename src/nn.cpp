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

    if (epoch % 20 == 0 && epoch != 0) {
      learningRate *= 0.5f;
    }
  }
  return 0;
}
