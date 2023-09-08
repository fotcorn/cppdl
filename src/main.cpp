#include <iostream>
#include <span>

#include "tensor.h"

#include "data/dataset.h"
#include "data/nn.h"

int main() {

  tensor<float> input = tensor<float>::matrix2d({{1.0f}, {2.0f}});

  // auto o1 = LAYER_0_WEIGHTS.matmul(input);
  // auto o2 = o1 + LAYER_0_BIASES;

  return 0;
}
