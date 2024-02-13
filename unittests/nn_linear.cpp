#include "nn.h"
#include <gtest/gtest.h>

class LinearLayerTest : public ::testing::Test {
protected:
  LinearLayer linearLayer{3, 2};
};

TEST_F(LinearLayerTest, Forward) {
  Tensor<float> input =
      Tensor<float>::matrix2d({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  Tensor<float> output = linearLayer.forward(input);
  ASSERT_EQ(output.shape, std::vector<size_t>({2, 2}));
}

TEST_F(LinearLayerTest, Backward) {
  Tensor<float> input = Tensor<float>::matrix2d({{1.0f, 2.0f, 3.0f},
                                                 {4.0f, 5.0f, 6.0f},
                                                 {4.0f, 5.0f, 6.0f},
                                                 {4.0f, 5.0f, 6.0f},
                                                 {4.0f, 5.0f, 6.0f}});
  Tensor<float> outGrad = Tensor<float>::matrix2d(
      {{0.5f, 2.0f}, {1.5f, 2.5f}, {1.5f, 2.5f}, {1.5f, 2.5f}, {1.5f, 2.5f}});
  linearLayer.forward(input);
  // linearLayer.backward(outGrad, input);

  ASSERT_EQ(linearLayer.biasGrad, Tensor<float>::matrix2d({{2.0f, 4.5f}}));
  ASSERT_EQ(
      linearLayer.weightGrad,
      Tensor<float>::matrix2d({{6.5f, 8.5f, 10.5f}, {12.0f, 16.5f, 21.0f}}));
}
