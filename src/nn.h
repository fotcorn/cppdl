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
    zeroGrad();
  }
  LinearLayer(Tensor<float> weight) : weight(std::move(weight)) { zeroGrad(); }
  LinearLayer(Tensor<float> weight, Tensor<float> bias)
      : weight(std::move(weight)), bias(std::move(bias)) {
    zeroGrad();
  }

  Tensor<float> forward(const Tensor<float> &input) {
    auto res1 = input.matmul(weight.transpose());
    auto res2 = res1 + bias;
    return res2;
  }

  Tensor<float> backward(const Tensor<float> &input,
                         const Tensor<float> &outGrad) {
    auto outGradSum =
        Tensor<float>::ones({1, outGrad.shape[0]}).matmul(outGrad);
    biasGrad = biasGrad + outGradSum;
    weightGrad = weightGrad + outGrad.transpose().matmul(input);
    return outGrad.matmul(weight);
  }

  void zeroGrad() {
    weightGrad = Tensor<float>(weight.shape, 0.0f);
    if (bias.size != 0) {
      biasGrad = Tensor<float>(bias.shape, 0.0f);
    }
  }

public:
  Tensor<float> weight;
  Tensor<float> bias;
  Tensor<float> biasGrad;
  Tensor<float> weightGrad;
};

class Tanh {
public:
  Tensor<float> forward(const Tensor<float> &input) {
    return input.apply([](float val) { return std::tanh(val); });
  }
  Tensor<float> backward(const Tensor<float> &input) {
    return input.apply([](float val) {
      float temp = std::tanh(val);
      return 1.0f - temp * temp;
    });
  }
};

class ReLU {
public:
  Tensor<float> forward(const Tensor<float> &input) {
    return input.apply([](float val) { return std::max<float>(0, val); });
  }
  Tensor<float> backward(const Tensor<float> &input,
                         const Tensor<float> &outGrad) {
    return outGrad *
           input.apply([](float val) { return val > 0.0f ? 1.0f : 0.0f; });
  }
};
