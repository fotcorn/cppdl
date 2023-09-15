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

// Elementwise tests.
TEST(Tensor, Add) {
  tensor<float> a = tensor<float>::vector({1.0f, 2.0f});
  tensor<float> b = a.add(1.0f);
  EXPECT_EQ(2.0f, b[0].item());
  EXPECT_EQ(3.0f, b[1].item());
}

TEST(Tensor, Sub) {
  tensor<float> a = tensor<float>::vector({2.0f, 3.0f});
  tensor<float> b = a.sub(1.0f);
  EXPECT_EQ(1.0f, b[0].item());
  EXPECT_EQ(2.0f, b[1].item());
}

TEST(Tensor, Mul) {
  tensor<float> a = tensor<float>::vector({2.0f, 3.0f});
  tensor<float> b = a.mul(2.0f);
  EXPECT_EQ(4.0f, b[0].item());
  EXPECT_EQ(6.0f, b[1].item());
}

TEST(Tensor, Div) {
  tensor<float> a = tensor<float>::vector({4.0f, 6.0f});
  tensor<float> b = a.div(2.0f);
  EXPECT_EQ(2.0f, b[0].item());
  EXPECT_EQ(3.0f, b[1].item());
}

// Broadcast tests
TEST(Tensor, VectorAddOneOne) {
  tensor<float> a = tensor<float>::vector({1.0f});
  tensor<float> b = tensor<float>::vector({3.5f});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({1}));
}

TEST(Tensor, VectorAddTwoOne) {
  tensor<float> a = tensor<float>::vector({1.0f, 2.0f});
  tensor<float> b = tensor<float>::vector({3.5f});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(5.5f, res[1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2}));
}

TEST(Tensor, VectorAddOneTwo) {
  tensor<float> a = tensor<float>::vector({3.5f});
  tensor<float> b = tensor<float>::vector({1.0f, 2.0f});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(5.5f, res[1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2}));
}

TEST(Tensor, VectorAddTwoTwo) {
  tensor<float> a = tensor<float>::vector({1.0f, 2.0f});
  tensor<float> b = tensor<float>::vector({3.5f, 4.5f});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(6.5f, res[1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2}));
}

TEST(Tensor, Matrix2DAdd2_V2) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}});
  tensor<float> b = tensor<float>::vector({3.5f, 4.5f});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({1, 2}));
}

TEST(Tensor, Matrix2DAdd22_V2) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::vector({3.5f, 4.5f});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(6.5f, res[1][0].item());
  EXPECT_EQ(8.5f, res[1][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix2DAdd2_M2) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f, 4.5f}});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({1, 2}));
}

TEST(Tensor, Matrix2DAdd22_M2) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f, 4.5f}});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(6.5f, res[1][0].item());
  EXPECT_EQ(8.5f, res[1][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix2DAdd2_22) {
  tensor<float> a = tensor<float>::matrix2d({{3.5f, 4.5f}});
  tensor<float> b = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(6.5f, res[1][0].item());
  EXPECT_EQ(8.5f, res[1][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix2DAdd22_22) {
  tensor<float> a = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  tensor<float> b = tensor<float>::matrix2d({{3.5f, 4.5f}, {5.5f, 6.5f}});
  tensor<float> res = a.add(b);
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(8.5f, res[1][0].item());
  EXPECT_EQ(10.5f, res[1][1].item());
  EXPECT_EQ(res.getShape(), std::vector<size_t>({2, 2}));
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
