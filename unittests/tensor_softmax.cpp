#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorSoftmax, Softmax1D) {
  Tensor<float> t = Tensor<float>::vector({1.0f, 2.0f, 3.0f, 4.0f});
  auto softmaxed = t.softmax();
  EXPECT_NEAR(softmaxed[0].item(), 0.0320586f, 1e-6);
  EXPECT_NEAR(softmaxed[1].item(), 0.0871443f, 1e-6);
  EXPECT_NEAR(softmaxed[2].item(), 0.2368828f, 1e-6);
  EXPECT_NEAR(softmaxed[3].item(), 0.6439143f, 1e-6);
}
