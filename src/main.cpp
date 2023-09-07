#include <iostream>
#include <span>

#include "tensor.h"

#include "data/dataset.h"
#include "data/nn.h"

int main() {
  tensor<float> t({2, 2});
  tensor o = tensor<float>::ones({2, 2});

  tensor<float> c = tensor<float>::constants({1.0f, 2.0f, 3.0f, 4.0f});
  tensor<float> x = tensor<float>::constants({3.0f});

  tensor<float> matrix = tensor<float>::constants({{1.0f, 2.0f}, {3.0f, 4.0f}});

  std::cout << c << "\n" << x << "\n" << matrix << std::endl;

  std::cout << matrix[0] << std::endl;
  std::cout << matrix[1] << std::endl;

  std::cout << matrix[1][0].item() << std::endl;

  std::cout << matrix.add(3.0f) << std::endl;

  /*
  // tensor operations
  tops::mul(t, o);
  tops::exp(t, 3);

  tops::softmax(t);
  tops::relu(t);

  tops::matmul(t1, t2);

  tops::reshape(t, {1,2,3});
  */

  return 0;
}
