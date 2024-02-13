#include "nn.h"
#include <gtest/gtest.h>

class ReLUTest : public ::testing::Test {
protected:
  ReLU relu;
};

TEST_F(ReLUTest, ForwardPositiveInput) {
  Tensor<float> input = Tensor<float>::vector({1.0f, 2.0f, 3.0f});
  Tensor<float> expectedOutput = Tensor<float>::vector({1.0f, 2.0f, 3.0f});
  ASSERT_EQ(relu.forward(input), expectedOutput);
}

TEST_F(ReLUTest, ForwardNegativeInput) {
  Tensor<float> input = Tensor<float>::vector({-1.0f, -2.0f, -3.0f});
  Tensor<float> expectedOutput = Tensor<float>::vector({0.0f, 0.0f, 0.0f});
  ASSERT_EQ(relu.forward(input), expectedOutput);
}

TEST_F(ReLUTest, BackwardPositiveGradient) {
  Tensor<float> input = Tensor<float>::vector({1.0f, 2.0f, 3.0f});
  Tensor<float> outGrad = Tensor<float>::vector({0.1f, 0.2f, 0.3f});
  Tensor<float> activations = relu.forward(input);
  Tensor<float> expectedOutput = Tensor<float>::vector({0.1f, 0.2f, 0.3f});
  // ASSERT_EQ(relu.backward(outGrad, activations), expectedOutput);
}

TEST_F(ReLUTest, BackwardNegativeGradient) {
  Tensor<float> input = Tensor<float>::vector({-1.0f, -2.0f, -3.0f});
  Tensor<float> outGrad = Tensor<float>::vector({0.1f, 0.2f, 0.3f});
  Tensor<float> activations = relu.forward(input);
  Tensor<float> expectedOutput = Tensor<float>::vector({0.0f, 0.0f, 0.0f});
  // ASSERT_EQ(relu.backward(outGrad, activations), expectedOutput);
}
