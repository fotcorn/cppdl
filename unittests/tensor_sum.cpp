#include <gtest/gtest.h>

#include "cppdl/tensor.h"

TEST(TensorSum, Sum1D) {
  Tensor<float> t = Tensor<float>::vector({1.0f, 2.0f, 3.0f, 4.0f});
  auto summed = t.sum();
  EXPECT_NEAR(summed[0].item(), 10.0f, 1e-6);
}

TEST(TensorSum, Sum2D_Dim0) {
  Tensor<float> t = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto summed = t.sum(0);
  EXPECT_NEAR(summed[0].item(), 4.0f, 1e-6);
  EXPECT_NEAR(summed[1].item(), 6.0f, 1e-6);
}

TEST(TensorSum, Sum2D_Dim1) {
  Tensor<float> t = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto summed = t.sum(1);
  EXPECT_NEAR(summed[0].item(), 3.0f, 1e-6);
  EXPECT_NEAR(summed[1].item(), 7.0f, 1e-6);
}
