#include <iostream>
#include <span>

#include "tensor.h"

int main() {
  // init
  tensor<float> t({2, 2});
  tensor o = tensor<float>::ones({2, 2});

  tensor<float> c = tensor<float>::constants({1.0f, 2.0f, 3.0f, 4.0f});
  tensor<float> x = tensor<float>::constants({3.0f});

  tensor<float> matrix = tensor<float>::constants({{1.0f, 2.0f}, {3.0f, 4.0f}});

  std::cout << c << "\n" << x << "\n" << matrix << std::endl;

  std::cout << matrix[0] << std::endl;
  std::cout << matrix[1] << std::endl;

  std::cout << matrix[1][0].item() << std::endl;

  std::cout << matrix.add(tensor<float>::constants({3.0f})) << std::endl;

  /*

  // tensor operations
  tops::mul(t, o);
  tops::exp(t, 3);

  tops::softmax(t);
  tops::relu(t);

  tops::matmul(t1, t2);

  tops::reshape(t, {1,2,3});

  // nn
  tensor<float> input({2, 1});

  tensor<float> l1w({16, 2});
  tensor<float> l1b({16, 1});

  tensor<float> l2w({16, 16});
  tensor<float> l2b({16, 1});

  tensor<float> l3w({1, 16});
  tensor<float> l3b({1, 1});
  */

  return 0;
}
