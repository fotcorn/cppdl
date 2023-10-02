#include <iostream>
#include <span>

#include "tensor.h"
#include "utils.h"

#include "data/dataset.h"
#include "data/nn.h"

template <typename Tensor>
Tensor net(Tensor input, Tensor w1, Tensor b1, Tensor w2, Tensor b2, Tensor w3,
           Tensor b3) {
  auto o1 = (w1.matmul(input) + b1).relu();
  auto o2 = (w2.matmul(o1) + b2).relu();
  return w3.matmul(o2) + b3;
}

int main() {
  tensor<float> input = tensor<float>::matrix2d(
      {{DATASET_VALUES[0][0].item()}, {DATASET_VALUES[0][1].item()}});
  auto result = net<tensor<float>>(input, LAYER_0_WEIGHTS, LAYER_0_BIASES,
                                   LAYER_1_WEIGHTS, LAYER_1_BIASES,
                                   LAYER_2_WEIGHTS, LAYER_2_BIASES);

  std::cout << result[0].item() << std::endl;
  std::cout << DATASET_LABELS[0].item() << std::endl;

  return 0;
}
