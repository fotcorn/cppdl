#include <gtest/gtest.h>

#include "tensor.h"

TEST(MeanSquareError, MSE_SingleDimension) {
  Tensor<float> a = Tensor<float>::vector({1.0f, 2.0f, 1.0f});
  Tensor<float> b = Tensor<float>::vector({1.5f, 2.0f, 3.5f});

  auto mse = a.meanSquareError(b);
  EXPECT_NEAR(mse, 2.166666, 1e-6);
}

TEST(MeanSquareError, MSE_TwoDimensions) {
  Tensor<float> a = Tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor<float> b = Tensor<float>::matrix2d({{1.5f, 2.0f}, {1.5f, 4.5f}});

  auto mse = a.meanSquareError(b);
  EXPECT_NEAR(mse, 0.6875, 1e-6);
}

TEST(MeanSquareError, MSE_MismatchedShapes) {
  Tensor<float> a = Tensor<float>::vector({1.0f, 2.0f, 3.0f});
  Tensor<float> b = Tensor<float>::matrix2d({{1.5f, 2.5f}, {3.5f, 4.5f}});

  EXPECT_THROW(
      {
        try {
          a.meanSquareError(b);
        } catch (const std::runtime_error &e) {
          EXPECT_STREQ("meanSquareError: shapes of operands do not match",
                       e.what());
          throw;
        }
      },
      std::runtime_error);
}
