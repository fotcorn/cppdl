#include <gtest/gtest.h>

#include "tensor.h"

TEST(Tensor, Subscript) {
  tensor<float> c = tensor<float>::constants({1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_EQ(1.0f, c[0].item());
  EXPECT_EQ(2.0f, c[1].item());
  EXPECT_EQ(3.0f, c[2].item());
  EXPECT_EQ(4.0f, c[3].item());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
