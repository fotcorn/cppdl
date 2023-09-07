#include <gtest/gtest.h>

#include "tensor.h"

TEST(Tensor, Subscript) {
  tensor<float> c = tensor<float>::constants({1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_EQ(1.0f, c[0].item());
  EXPECT_EQ(2.0f, c[1].item());
  EXPECT_EQ(3.0f, c[2].item());
  EXPECT_EQ(4.0f, c[3].item());
}

TEST(Tensor, Add) {
  tensor<float> a = tensor<float>::constants({1.0f, 2.0f});
  tensor<float> b = a.add(1.0f);
  EXPECT_EQ(2.0f, b[0].item());
  EXPECT_EQ(3.0f, b[1].item());
}

TEST(Tensor, Sub) {
  tensor<float> a = tensor<float>::constants({2.0f, 3.0f});
  tensor<float> b = a.sub(1.0f);
  EXPECT_EQ(1.0f, b[0].item());
  EXPECT_EQ(2.0f, b[1].item());
}

TEST(Tensor, Mul) {
  tensor<float> a = tensor<float>::constants({2.0f, 3.0f});
  tensor<float> b = a.mul(2.0f);
  EXPECT_EQ(4.0f, b[0].item());
  EXPECT_EQ(6.0f, b[1].item());
}

TEST(Tensor, Div) {
  tensor<float> a = tensor<float>::constants({4.0f, 6.0f});
  tensor<float> b = a.div(2.0f);
  EXPECT_EQ(2.0f, b[0].item());
  EXPECT_EQ(3.0f, b[1].item());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
