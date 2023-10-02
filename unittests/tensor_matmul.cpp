#include <gtest/gtest.h>

#include "tensor.h"

TEST(MatMul, M12_21) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f}, {4.5f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({1, 1}));
  EXPECT_EQ(res[0][0].item(), 12.5f);
}

TEST(MatMul, M21_12) {
  tensor<float> a = tensor<float>::matrix2d({{3.5f}, {4.5f}});
  tensor<float> b = tensor<float>::matrix2d({{1.0f, 2.0f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
  EXPECT_EQ(res[0][0].item(), 3.5f);
  EXPECT_EQ(res[0][1].item(), 7.0f);
  EXPECT_EQ(res[1][0].item(), 4.5f);
  EXPECT_EQ(res[1][1].item(), 9.0f);
}

TEST(MatMul, M22_21) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f}, {4.5f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 1}));
  EXPECT_EQ(res[0][0].item(), 12.5f);
  EXPECT_EQ(res[1][0].item(), 28.5f);
}

TEST(MatMul, M22_22) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f, 4.5f}, {5.5f, 6.5f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
  EXPECT_EQ(res[0][0].item(), 14.5f);
  EXPECT_EQ(res[0][1].item(), 17.5f);
  EXPECT_EQ(res[1][0].item(), 32.5f);
  EXPECT_EQ(res[1][1].item(), 39.5f);
}

TEST(MatMul, M32_22) {
  tensor<float> a =
      tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f, 4.5f}, {5.5f, 6.5f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({3, 2}));
  EXPECT_EQ(res[0][0].item(), 14.5f);
  EXPECT_EQ(res[0][1].item(), 17.5f);
  EXPECT_EQ(res[1][0].item(), 32.5f);
  EXPECT_EQ(res[1][1].item(), 39.5f);
  EXPECT_EQ(res[2][0].item(), 50.5f);
  EXPECT_EQ(res[2][1].item(), 61.5f);
}

TEST(MatMul, M32_21) {
  tensor<float> a =
      tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f}, {4.5f}});

  auto res = a.matmul(b);
  EXPECT_EQ(res.getShape(), std::vector<size_t>({3, 1}));
  EXPECT_EQ(res[0][0].item(), 12.5f);
  EXPECT_EQ(res[1][0].item(), 28.5f);
  EXPECT_EQ(res[2][0].item(), 44.5f);
}
