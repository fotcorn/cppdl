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

int main() { return 0; }
