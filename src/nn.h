#include "tensor.h"
#include <fmt/format.h>
#include <fmt/ranges.h>

class LinearLayer {
public:
  LinearLayer(size_t numInputs, size_t numOutputs, bool hasBias = true)
      : weight(tensor<float>::random({numInputs, numOutputs})) {
    if (hasBias) {
      bias = tensor<float>::random({numOutputs});
    }
  }
  LinearLayer(tensor<float> weight) : weight(weight) {}
  LinearLayer(tensor<float> weight, tensor<float> bias)
      : weight(weight), bias(bias) {}

  tensor<float> forward(tensor<float> input) {
    activations = input.matmul(weight) + bias;
    return activations;
  }

private:
  tensor<float> weight;
  tensor<float> bias;
  tensor<float> activations;
  tensor<float> biasGrad;
  tensor<float> weightGrad;
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
