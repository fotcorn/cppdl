#include <gtest/gtest.h>

#include "cppdl/tensor.h"

TEST(Tensor, VectorAddOneOne) {
  Tensor<float> a = Tensor<float>::vector({1.0f});
  Tensor<float> b = Tensor<float>::vector({3.5f});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({1}));
}

TEST(Tensor, VectorAddTwoOne) {
  Tensor<float> a = Tensor<float>::vector({1.0f, 2.0f});
  Tensor<float> b = Tensor<float>::vector({3.5f});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(5.5f, res[1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2}));
}

TEST(Tensor, VectorAddOneTwo) {
  Tensor<float> a = Tensor<float>::vector({3.5f});
  Tensor<float> b = Tensor<float>::vector({1.0f, 2.0f});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(5.5f, res[1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2}));
}

TEST(Tensor, VectorAddTwoTwo) {
  Tensor<float> a = Tensor<float>::vector({1.0f, 2.0f});
  Tensor<float> b = Tensor<float>::vector({3.5f, 4.5f});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0].item());
  EXPECT_EQ(6.5f, res[1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2}));
}

TEST(Tensor, Matrix2DAdd2_V2) {
  Tensor<float> a = Tensor<float>::matrix2d({{1.0f, 2.0f}});
  Tensor<float> b = Tensor<float>::vector({3.5f, 4.5f});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({1, 2}));
}

TEST(Tensor, Matrix2DAdd22_V2) {
  Tensor<float> a = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor<float> b = Tensor<float>::vector({3.5f, 4.5f});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(6.5f, res[1][0].item());
  EXPECT_EQ(8.5f, res[1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix2DAdd2_M2) {
  Tensor<float> a = Tensor<float>::matrix2d({{1.0f, 2.0f}});
  Tensor<float> b = Tensor<float>::matrix2d({{3.5f, 4.5f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({1, 2}));
}

TEST(Tensor, Matrix2DAdd22_M2) {
  Tensor<float> a = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor<float> b = Tensor<float>::matrix2d({{3.5f, 4.5f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(6.5f, res[1][0].item());
  EXPECT_EQ(8.5f, res[1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix2DAdd2_22) {
  Tensor<float> a = Tensor<float>::matrix2d({{3.5f, 4.5f}});
  Tensor<float> b = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(6.5f, res[1][0].item());
  EXPECT_EQ(8.5f, res[1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix2DAdd22_22) {
  Tensor<float> a = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor<float> b = Tensor<float>::matrix2d({{3.5f, 4.5f}, {5.5f, 6.5f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(4.5f, res[0][0].item());
  EXPECT_EQ(6.5f, res[0][1].item());
  EXPECT_EQ(8.5f, res[1][0].item());
  EXPECT_EQ(10.5f, res[1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2}));
}

TEST(Tensor, Matrix3DAdd_1_22) {
  Tensor<float> a = Tensor<float>::stack(
      {Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}}),
       Tensor<float>::matrix2d({{5.0f, 6.0f}, {7.0f, 8.0f}})});
  Tensor<float> b = Tensor<float>::matrix2d({{1.0f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(2.0f, res[0][0][0].item());
  EXPECT_EQ(3.0f, res[0][0][1].item());
  EXPECT_EQ(4.0f, res[0][1][0].item());
  EXPECT_EQ(5.0f, res[0][1][1].item());
  EXPECT_EQ(6.0f, res[1][0][0].item());
  EXPECT_EQ(7.0f, res[1][0][1].item());
  EXPECT_EQ(8.0f, res[1][1][0].item());
  EXPECT_EQ(9.0f, res[1][1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2, 2}));
}

TEST(Tensor, Matrix3DAdd_12_22) {
  Tensor<float> a = Tensor<float>::stack(
      {Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}}),
       Tensor<float>::matrix2d({{5.0f, 6.0f}, {7.0f, 8.0f}})});
  Tensor<float> b = Tensor<float>::matrix2d({{1.0f, 2.0f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(2.0f, res[0][0][0].item());
  EXPECT_EQ(4.0f, res[0][0][1].item());
  EXPECT_EQ(4.0f, res[0][1][0].item());
  EXPECT_EQ(6.0f, res[0][1][1].item());
  EXPECT_EQ(6.0f, res[1][0][0].item());
  EXPECT_EQ(8.0f, res[1][0][1].item());
  EXPECT_EQ(8.0f, res[1][1][0].item());
  EXPECT_EQ(10.0f, res[1][1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2, 2}));
}

TEST(Tensor, Matrix3DAdd_22_22) {
  Tensor<float> a = Tensor<float>::stack(
      {Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}}),
       Tensor<float>::matrix2d({{5.0f, 6.0f}, {7.0f, 8.0f}})});
  Tensor<float> b = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor<float> res = a + b;
  EXPECT_EQ(2.0f, res[0][0][0].item());
  EXPECT_EQ(4.0f, res[0][0][1].item());
  EXPECT_EQ(6.0f, res[0][1][0].item());
  EXPECT_EQ(8.0f, res[0][1][1].item());
  EXPECT_EQ(6.0f, res[1][0][0].item());
  EXPECT_EQ(8.0f, res[1][0][1].item());
  EXPECT_EQ(10.0f, res[1][1][0].item());
  EXPECT_EQ(12.0f, res[1][1][1].item());
  EXPECT_EQ(res.shape, std::vector<size_t>({2, 2, 2}));
}
