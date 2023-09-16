#include <gtest/gtest.h>

#include "tensor.h"

TEST(Tensor, Subscript) {
  tensor<float> c = tensor<float>::vector({1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_EQ(1.0f, c[0].item());
  EXPECT_EQ(2.0f, c[1].item());
  EXPECT_EQ(3.0f, c[2].item());
  EXPECT_EQ(4.0f, c[3].item());
}

TEST(Tensor, Constants2D) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  EXPECT_EQ(1.0f, a[0][0].item());
  EXPECT_EQ(2.0f, a[0][1].item());
  EXPECT_EQ(3.0f, a[1][0].item());
  EXPECT_EQ(4.0f, a[1][1].item());
}
