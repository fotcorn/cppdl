#include "nn.h"
#include <gtest/gtest.h>

class ReLUTest : public ::testing::Test {
protected:
  ReLU relu;
};

TEST_F(ReLUTest, ForwardPositiveInput) {
  tensor<float> input = tensor<float>::vector({1.0f, 2.0f, 3.0f});
  tensor<float> expected_output = tensor<float>::vector({1.0f, 2.0f, 3.0f});
  ASSERT_EQ(relu.forward(input), expected_output);
}

TEST_F(ReLUTest, ForwardNegativeInput) {
  tensor<float> input = tensor<float>::vector({-1.0f, -2.0f, -3.0f});
  tensor<float> expected_output = tensor<float>::vector({0.0f, 0.0f, 0.0f});
  ASSERT_EQ(relu.forward(input), expected_output);
}

TEST_F(ReLUTest, BackwardPositiveGradient) {
  tensor<float> input = tensor<float>::vector({1.0f, 2.0f, 3.0f});
  tensor<float> outGrad = tensor<float>::vector({0.1f, 0.2f, 0.3f});
  relu.forward(input);
  tensor<float> expected_output = tensor<float>::vector({0.1f, 0.2f, 0.3f});
  ASSERT_EQ(relu.backward(outGrad), expected_output);
}

TEST_F(ReLUTest, BackwardNegativeGradient) {
  tensor<float> input = tensor<float>::vector({-1.0f, -2.0f, -3.0f});
  tensor<float> outGrad = tensor<float>::vector({0.1f, 0.2f, 0.3f});
  relu.forward(input);
  tensor<float> expected_output = tensor<float>::vector({0.0f, 0.0f, 0.0f});
  ASSERT_EQ(relu.backward(outGrad), expected_output);
}
