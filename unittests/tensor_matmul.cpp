#include <gtest/gtest.h>

#include "tensor.h"

TEST(MatMul, M22_12) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f}, {4.5f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 1}));
  EXPECT_EQ(res[0][0].item(), 12.5f);
  EXPECT_EQ(res[1][0].item(), 28.5f);
}