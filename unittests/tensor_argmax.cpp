#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorArgmax, Argmax1D) {
  Tensor<float> t = Tensor<float>::vector({1.0f, 3.0f, 2.0f, 4.0f});
  auto argmax = t.argmax(0);
  EXPECT_EQ(argmax.item(), 3);
}

TEST(TensorArgmax, Argmax2D_Dim0) {
  Tensor<float> t =
      Tensor<float>::matrix2d({{3.0f, 2.0f}, {1.0f, 4.0f}, {5.0f, 1.0f}});
  auto argmax = t.argmax(0);
  EXPECT_EQ(argmax.shape, std::vector<size_t>({2}));
  EXPECT_EQ(argmax[0].item(), 2);
  EXPECT_EQ(argmax[1].item(), 1);
}

TEST(TensorArgmax, Argmax2D_Dim1) {
  Tensor<float> t =
      Tensor<float>::matrix2d({{1.0f, 4.0f}, {3.0f, 2.0f}, {5.0f, 6.0f}});
  auto argmax = t.argmax(1);
  EXPECT_EQ(argmax.shape, std::vector<size_t>({3}));
  EXPECT_EQ(argmax[0].item(), 1);
  EXPECT_EQ(argmax[1].item(), 0);
  EXPECT_EQ(argmax[2].item(), 1);
}
