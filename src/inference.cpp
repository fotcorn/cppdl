#include <cmath>
#include <iostream>
#include <span>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "tensor.h"

#include "data/dataset.h"
#include "data/nn.h"

template <typename Tensor>
Tensor net(Tensor input, Tensor w1, Tensor b1, Tensor w2, Tensor b2, Tensor w3,
           Tensor b3) {
  auto o1 = (input.matmul(w1.transpose()) + b1).relu();
  auto o2 = (o1.matmul(w2.transpose()) + b2).relu();
  return o2.matmul(w3) + b3;
}

int main() {
  auto result = net<tensor<float>>(
      DATASET_VALUES, LAYER_0_WEIGHTS, LAYER_0_BIASES, LAYER_1_WEIGHTS,
      LAYER_1_BIASES, LAYER_2_WEIGHTS, LAYER_2_BIASES);

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
