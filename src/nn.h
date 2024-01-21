#pragma once

#include "tensor.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

class LinearLayer {
public:
  LinearLayer(size_t numInputs, size_t numOutputs, bool hasBias = true)
      : weight(Tensor<float>::random({numOutputs, numInputs})) {
    if (hasBias) {
      bias = Tensor<float>::random({numOutputs});
    }
    initGrad();
  }
  LinearLayer(Tensor<float> weight) : weight(std::move(weight)) { initGrad(); }
  LinearLayer(Tensor<float> weight, Tensor<float> bias)
      : weight(std::move(weight)), bias(std::move(bias)) {
    initGrad();
  }

  Tensor<float> forward(const Tensor<float> &input) {
    return weight.matmul(input).transpose() + bias;
  }

  Tensor<float> backward(const Tensor<float> &outGrad,
                         const Tensor<float> &input) {
    biasGrad = biasGrad + outGrad.transpose();
    auto inGrad = outGrad.matmul(input.transpose());
    weightGrad = weightGrad + inGrad;
    return inGrad;
  }

private:
  void initGrad() {
    weightGrad = Tensor<float>(weight.shape, 0.0f);
    if (bias.size != 0) {
      biasGrad = Tensor<float>(bias.shape, 0.0f);
    }
  }

  Tensor<float> weight;
  Tensor<float> bias;

public:
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
