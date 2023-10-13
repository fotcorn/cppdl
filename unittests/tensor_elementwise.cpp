#include <gtest/gtest.h>

#include "tensor.h"

TEST(Tensor, Add) {
  tensor<float> a = tensor<float>::vector({1.0f, 2.0f});
  tensor<float> b = a + 1.0f;
  EXPECT_EQ(2.0f, b[0].item());
  EXPECT_EQ(3.0f, b[1].item());
}

TEST(Tensor, Sub) {
  tensor<float> a = tensor<float>::vector({2.0f, 3.0f});
  tensor<float> b = a - 1.0f;
  EXPECT_EQ(1.0f, b[0].item());
  EXPECT_EQ(2.0f, b[1].item());
}

TEST(Tensor, Mul) {
  tensor<float> a = tensor<float>::vector({2.0f, 3.0f});
  tensor<float> b = a * 2.0f;
  EXPECT_EQ(4.0f, b[0].item());
  EXPECT_EQ(6.0f, b[1].item());
}

TEST(Tensor, Div) {
  tensor<float> a = tensor<float>::vector({4.0f, 6.0f});
  tensor<float> b = a / 2.0f;
  EXPECT_EQ(2.0f, b[0].item());
  EXPECT_EQ(3.0f, b[1].item());
}

TEST(Tensor, ReLU) {
  tensor<float> a = tensor<float>::matrix2d({{-1.0f, 0.0f}, {3.0f, -4.0f}});
  tensor<float> res = a.relu();
  EXPECT_EQ(0.0f, res[0][0].item());
  EXPECT_EQ(0.0f, res[0][1].item());
  EXPECT_EQ(3.0f, res[1][0].item());
  EXPECT_EQ(0.0f, res[1][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
}

TEST(Tensor, ReLU3D) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, -2.0f}, {3.0f, -4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{-1.0f, 2.0f}, {-3.0f, 4.0f}});
  tensor<float> c = tensor<float>::stack({a, b}).relu();
  EXPECT_EQ(1.0f, c[0][0][0].item());
  EXPECT_EQ(0.0f, c[0][0][1].item());
  EXPECT_EQ(3.0f, c[0][1][0].item());
  EXPECT_EQ(0.0f, c[0][1][1].item());
  EXPECT_EQ(0.0f, c[1][0][0].item());
  EXPECT_EQ(2.0f, c[1][0][1].item());
  EXPECT_EQ(0.0f, c[1][1][0].item());
  EXPECT_EQ(4.0f, c[1][1][1].item());
  EXPECT_EQ(c.getShape(), std::vector<size_t>({2, 2, 2}));
}

TEST(Tensor, Add3D) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{5.0f, 6.0f}, {7.0f, 8.0f}});
  tensor<float> c = tensor<float>::stack({a, b});
  tensor<float> d = c + 1.0f;
  EXPECT_EQ(2.0f, d[0][0][0].item());
  EXPECT_EQ(3.0f, d[0][0][1].item());
  EXPECT_EQ(4.0f, d[0][1][0].item());
  EXPECT_EQ(5.0f, d[0][1][1].item());
  EXPECT_EQ(6.0f, d[1][0][0].item());
  EXPECT_EQ(7.0f, d[1][0][1].item());
  EXPECT_EQ(8.0f, d[1][1][0].item());
  EXPECT_EQ(9.0f, d[1][1][1].item());
  EXPECT_EQ(d.getShape(), std::vector<size_t>({2, 2, 2}));
}
