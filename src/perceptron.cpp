#include "tensor.h"

#include <iostream>

tensor<int> toIntTensor(const tensor<float> &input) {
  tensor<int> output(input.shape);
  for (size_t i = 0; i < input.size; i++) {
    output.data.get()[i] = round(input.data.get()[i]);
  }
  return output;
}

int main() {
  auto data = tensor<float>::matrix2d({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
  auto orGroudTruth = tensor<float>::vector({0, 1, 1, 1});
  auto andGroudTruth = tensor<float>::vector({0, 0, 0, 1});

  auto orWeights = tensor<float>::matrix2d({{0.5}, {0.5}});
  auto andWeights = tensor<float>::matrix2d({{0.25}, {0.25}});

  std::cout << toIntTensor(data.matmul(orWeights)) << std::endl;
  std::cout << toIntTensor(data.matmul(andWeights)) << std::endl;

  return 0;
}
