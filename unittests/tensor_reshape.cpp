#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorReshape, ShapesDoNotMatch) {
  tensor<float> t({2, 3});
  EXPECT_THROW(t.reshape({2, 2}), std::runtime_error);
}

TEST(TensorReshape, NewShapesApplied) {
  tensor<float> t =
      tensor<float>::matrix2d({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  auto reshaped = t.reshape({3, 2});
  EXPECT_EQ(reshaped.getShape(), std::vector<size_t>({3, 2}));
  EXPECT_EQ(reshaped[0][0].item(), 1.0f);
  EXPECT_EQ(reshaped[0][1].item(), 2.0f);
  EXPECT_EQ(reshaped[1][0].item(), 3.0f);
  EXPECT_EQ(reshaped[1][1].item(), 4.0f);
  EXPECT_EQ(reshaped[2][0].item(), 5.0f);
  EXPECT_EQ(reshaped[2][1].item(), 6.0f);
}

TEST(TensorReshape, OneDtoTwoD) {
  tensor<float> t = tensor<float>::vector({1.0f, 2.0f, 3.0f, 4.0f});
  auto reshaped = t.reshape({2, 2});
  EXPECT_EQ(reshaped.getShape(), std::vector<size_t>({2, 2}));
  EXPECT_EQ(reshaped[0][0].item(), 1.0f);
  EXPECT_EQ(reshaped[0][1].item(), 2.0f);
  EXPECT_EQ(reshaped[1][0].item(), 3.0f);
  EXPECT_EQ(reshaped[1][1].item(), 4.0f);
}

TEST(TensorReshape, TwoDtoOneD) {
  tensor<float> t = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto reshaped = t.reshape({4});
  EXPECT_EQ(reshaped.getShape(), std::vector<size_t>({4}));
  EXPECT_EQ(reshaped[0].item(), 1.0f);
  EXPECT_EQ(reshaped[1].item(), 2.0f);
  EXPECT_EQ(reshaped[2].item(), 3.0f);
  EXPECT_EQ(reshaped[3].item(), 4.0f);
}
