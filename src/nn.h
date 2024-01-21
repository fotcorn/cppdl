#pragma once

#include "tensor.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

class LinearLayer {
public:
  LinearLayer(size_t numInputs, size_t numOutputs, bool hasBias = true)
      : weight(Tensor<float>::random({numInputs, numOutputs})) {
    if (hasBias) {
      bias = Tensor<float>::random({numOutputs});
    }
  }
  LinearLayer(Tensor<float> weight) : weight(std::move(weight)) {}
  LinearLayer(Tensor<float> weight, Tensor<float> bias)
      : weight(std::move(weight)), bias(std::move(bias)) {}

  Tensor<float> forward(const Tensor<float> &input) {
    activations = input.matmul(weight) + bias;
    return activations;
  }

private:
  Tensor<float> weight;
  Tensor<float> bias;
  Tensor<float> activations;
  Tensor<float> biasGrad;
  Tensor<float> weightGrad;
};

class Tanh {
public:
  Tensor<float> forward(const Tensor<float> &input) {
    activations = input.apply([](float val) { return std::tanh(val); });
    return activations;
  }
  Tensor<float> backward(const Tensor<float> &outGrad) {
    auto res = activations.apply([](float val) {
      float temp = std::tanh(val);
      return 1.0f - temp * temp;
    });
    return res * outGrad;
  }

private:
  Tensor<float> activations;
};

class ReLU {
public:
  Tensor<float> forward(const Tensor<float> &input) {
    activations =
        input.apply([](float val) { return std::max<float>(0, val); });
    return activations;
  }
  Tensor<float> backward(const Tensor<float> &outGrad) {
    auto res =
        activations.apply([](float val) { return val > 0.0f ? 1.0f : 0.0f; });
    return res * outGrad;
  }

private:
  Tensor<float> activations;
};
