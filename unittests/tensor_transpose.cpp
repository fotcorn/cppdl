#include <gtest/gtest.h>

#include "tensor.h"

TEST(Transpose, Vector1D) {
  tensor<float> t = tensor<float>::vector({1.0f, 2.0f, 3.0f});
  auto transposed = t.transpose();
  EXPECT_EQ(transposed.getShape(), std::vector<size_t>({3}));
  EXPECT_EQ(transposed[0].item(), 1.0f);
  EXPECT_EQ(transposed[1].item(), 2.0f);
  EXPECT_EQ(transposed[2].item(), 3.0f);
}

TEST(Transpose, Matrix2D) {
  tensor<float> t = tensor<float>::matrix2d({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto transposed = t.transpose();
  EXPECT_EQ(transposed.getShape(), std::vector<size_t>({2, 2}));
  EXPECT_EQ(transposed[0][0].item(), 1.0f);
  EXPECT_EQ(transposed[0][1].item(), 3.0f);
  EXPECT_EQ(transposed[1][0].item(), 2.0f);
  EXPECT_EQ(transposed[1][1].item(), 4.0f);
}

TEST(Transpose, Tensor3D) {
  tensor<float> t1 = tensor<float>::matrix2d({{1.0f, 2.0f, 3.0f, 4.0f},
                                              {5.0f, 6.0f, 7.0f, 8.0f},
                                              {9.0f, 10.0f, 11.0f, 12.0f}});
  tensor<float> t2 = tensor<float>::matrix2d({{13.0f, 14.0f, 15.0f, 16.0f},
                                              {17.0f, 18.0f, 19.0f, 20.0f},
                                              {21.0f, 22.0f, 23.0f, 24.0f}});
  tensor<float> t = tensor<float>::stack({t1, t2});
  auto transposed = t.transpose();
  EXPECT_EQ(t.getShape(), std::vector<size_t>({2, 3, 4}));
  EXPECT_EQ(transposed.getShape(), std::vector<size_t>({4, 3, 2}));
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 2; k++) {
        EXPECT_EQ(transposed[i][j][k].item(), t[k][j][i].item());
      }
    }
  }
}
