#include <gtest/gtest.h>

#include "tensor.h"

TEST(Tensor, Stack) {
  tensor<float> a = tensor<float>::vector({1.0f, 2.0f, 3.0f});
  tensor<float> b = tensor<float>::vector({4.0f, 5.0f, 6.0f});
  tensor<float> c = tensor<float>::stack({a, b});

  EXPECT_EQ(1.0f, c[0][0].item());
  EXPECT_EQ(2.0f, c[0][1].item());
  EXPECT_EQ(3.0f, c[0][2].item());
  EXPECT_EQ(4.0f, c[1][0].item());
  EXPECT_EQ(5.0f, c[1][1].item());
  EXPECT_EQ(6.0f, c[1][2].item());
  EXPECT_EQ(c.getShape(), std::vector<size_t>({2, 3}));
}
