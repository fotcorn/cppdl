#include <gtest/gtest.h>

#include "tensor.h"

TEST(Slice, M22) {
  tensor<float> t = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});

  auto s1 = t[0];
  EXPECT_EQ(s1.getShape(), std::vector<size_t>({2}));
  EXPECT_EQ(s1[0].item(), 1.0f);
  EXPECT_EQ(s1[0].getShape(), std::vector<size_t>({1}));
  EXPECT_EQ(s1[1].item(), 2.0f);
  EXPECT_EQ(s1[1].getShape(), std::vector<size_t>({1}));

  auto s2 = t[1];
  EXPECT_EQ(s2.getShape(), std::vector<size_t>({2}));
  EXPECT_EQ(s2[0].item(), 3.0f);
  EXPECT_EQ(s2[0].getShape(), std::vector<size_t>({1}));
  EXPECT_EQ(s2[1].item(), 4.0f);
  EXPECT_EQ(s2[1].getShape(), std::vector<size_t>({1}));
}

TEST(Slice, M22_Inline) {
  tensor<float> t = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_EQ(t[0].getShape(), std::vector<size_t>({2}));
  EXPECT_EQ(t[0][0].item(), 1.0f);
  EXPECT_EQ(t[0][0].getShape(), std::vector<size_t>({1}));
  EXPECT_EQ(t[0][1].item(), 2.0f);
  EXPECT_EQ(t[0][1].getShape(), std::vector<size_t>({1}));

  EXPECT_EQ(t[1].getShape(), std::vector<size_t>({2}));
  EXPECT_EQ(t[1][0].item(), 3.0f);
  EXPECT_EQ(t[1][0].getShape(), std::vector<size_t>({1}));
  EXPECT_EQ(t[1][1].item(), 4.0f);
  EXPECT_EQ(t[1][1].getShape(), std::vector<size_t>({1}));
}

TEST(Slice, OutOfBounds) {
  tensor<float> t = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});

  EXPECT_THROW(t[2], std::runtime_error);
  EXPECT_THROW(t[0][2], std::runtime_error);
}
