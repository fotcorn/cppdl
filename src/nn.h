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
    auto res1 = input.matmul(weight.transpose());
    auto res2 = res1 + bias;
    return res2;
  }

  /*Tensor<float> backward(const Tensor<float> &outGrad,
                         const Tensor<float> &input) {
    biasGrad =
    biasGrad + Tensor<float>::ones({1, outGrad.shape[0]}).matmul(outGrad);
    auto inGrad = outGrad * input;

    auto wg = outGrad * input;
    auto ig = outGrad * weight;

    weightGrad =
    weightGrad + Tensor<float>::ones({1, inGrad.shape[0]}).matmul(inGrad);
    return inGrad;
  }*/

private:
  void initGrad() {
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
  Tensor<float> backward(const Tensor<float> &outGrad,
                         const Tensor<float> &activations) {
    auto res = activations.apply([](float val) {
      float temp = std::tanh(val);
      return 1.0f - temp * temp;
    });
    return res * outGrad;
  }
};

class ReLU {
public:
  Tensor<float> forward(const Tensor<float> &input) {
    return input.apply([](float val) { return std::max<float>(0, val); });
  }
  Tensor<float> backward(const Tensor<float> &activations) {
    return activations.apply(
        [](float val) { return val > 0.0f ? 1.0f : 0.0f; });
  }
};
