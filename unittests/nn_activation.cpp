#include "nn.h"
#include <gtest/gtest.h>

TEST(ActivationsTest, ReLU) {
  ReLU relu;
  Tensor<float> input = Tensor<float>::vector({1.0f, -2.0f, 3.0f});
  Tensor<float> expectedOutput = Tensor<float>::vector({1.0f, 0.0f, 3.0f});
  ASSERT_EQ(relu.forward(input), expectedOutput);
}
