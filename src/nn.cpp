#include "data/nn.h"
#include "data/dataset.h"
#include "tensor.h"

class LinearLayer {
public:
  LinearLayer(size_t numInputs, size_t numOutputs, bool hasBias = true)
      : weights(tensor<float>::random({numInputs, numOutputs})) {
    if (hasBias) {
      biases = tensor<float>::random({numOutputs});
    }
  }
  LinearLayer(tensor<float> weights) : weights(weights) {}
  LinearLayer(tensor<float> weights, tensor<float> biases)
      : weights(weights), biases(biases) {}

  tensor<float> forward(tensor<float> input) {
    activations = input.matmul(weights) + biases;
    return activations;
  }

private:
  tensor<float> weights;
  tensor<float> biases;
  tensor<float> activations;
};

class Tanh {
public:
  tensor<float> forward(tensor<float> input) {
    activations = input.apply([](float val) { return std::tanh(val); });
    return activations;
  }
  tensor<float> backward(tensor<float> outGrad) {
    auto res = activations.apply([](float val) {
      float temp = std::tanh(val);
      return 1.0f - temp * temp;
    });
    return res * outGrad;
  }

private:
  tensor<float> activations;
};

class ReLU {
public:
  tensor<float> forward(tensor<float> input) {
    activations =
        input.apply([](float val) { return std::max<float>(0, val); });
    return activations;
  }
  tensor<float> backward(tensor<float> outGrad) {
    auto res =
        activations.apply([](float val) { return val > 0.0f ? 1.0f : 0.0f; });
    return res * outGrad;
  }

private:
  tensor<float> activations;
};

int main() {
  LinearLayer layer0(LAYER_0_WEIGHTS.transpose(), LAYER_0_BIASES);
  ReLU layer0Activation;
  LinearLayer layer1(LAYER_1_WEIGHTS.transpose(), LAYER_1_BIASES);
  ReLU layer1Activation;
  LinearLayer layer2(LAYER_2_WEIGHTS, LAYER_2_BIASES);

  auto r1 = layer0.forward(DATASET_VALUES);
  auto r2 = layer0Activation.forward(r1);
  auto r3 = layer1.forward(r2);
  auto r4 = layer1Activation.forward(r3);
  auto result = layer2.forward(r4);

  int correct = 0;
  for (size_t i = 0; i < DATASET_VALUES.getShape()[0]; i++) {
    if (std::signbit(result[i].item()) ==
        std::signbit(DATASET_LABELS[i].item())) {
      correct++;
    }
  }

  auto flatResult = result.reshape({100});

  float accuracy = static_cast<float>(correct) / DATASET_VALUES.getShape()[0];
  fmt::println("Accuracy: {}", accuracy);

  float mse = flatResult.meanSquareError(DATASET_LABELS);
  fmt::println("MSE: {}", mse);

  return 0;
}
