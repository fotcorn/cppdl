#include <gtest/gtest.h>

#include "cppdl/tensor.h"

TEST(Slice, M22) {
  Tensor<float> t = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});

  auto s1 = t[0];
  EXPECT_EQ(s1.shape, std::vector<size_t>({2}));
  EXPECT_EQ(s1[0].item(), 1.0f);
  EXPECT_EQ(s1[0].shape, std::vector<size_t>({1}));
  EXPECT_EQ(s1[1].item(), 2.0f);
  EXPECT_EQ(s1[1].shape, std::vector<size_t>({1}));

  auto s2 = t[1];
  EXPECT_EQ(s2.shape, std::vector<size_t>({2}));
  EXPECT_EQ(s2[0].item(), 3.0f);
  EXPECT_EQ(s2[0].shape, std::vector<size_t>({1}));
  EXPECT_EQ(s2[1].item(), 4.0f);
  EXPECT_EQ(s2[1].shape, std::vector<size_t>({1}));
}

TEST(Slice, M22_Inline) {
  Tensor<float> t = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_EQ(t[0].shape, std::vector<size_t>({2}));
  EXPECT_EQ(t[0][0].item(), 1.0f);
  EXPECT_EQ(t[0][0].shape, std::vector<size_t>({1}));
  EXPECT_EQ(t[0][1].item(), 2.0f);
  EXPECT_EQ(t[0][1].shape, std::vector<size_t>({1}));

  EXPECT_EQ(t[1].shape, std::vector<size_t>({2}));
  EXPECT_EQ(t[1][0].item(), 3.0f);
  EXPECT_EQ(t[1][0].shape, std::vector<size_t>({1}));
  EXPECT_EQ(t[1][1].item(), 4.0f);
  EXPECT_EQ(t[1][1].shape, std::vector<size_t>({1}));
}

TEST(Slice, OutOfBounds) {
  Tensor<float> t = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_THROW(t[2], std::runtime_error);
  EXPECT_THROW(t[0][2], std::runtime_error);
}
