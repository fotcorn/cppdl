#include "cppdl/mnist_utils.h"
#include "cppdl/nn.h"
#include "cppdl/serialization.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fmt::println(stderr, "Usage: {} <weights_file>\n", argv[0]);
    return 1;
  }

  std::string weightsFile = argv[1];
  std::ifstream file(weightsFile, std::ios::binary);
  if (!file.is_open()) {
    fmt::println(stderr, "Error opening weights file: {}\n", weightsFile);
    return 1;
  }

  if (!readFileHeader(file)) {
    fmt::println(stderr, "Invalid weights file format.\n");
    return 1;
  }

  // Load validation dataset
  auto rawValidationImages = loadImages("../data/mnist/t10k-images-idx3-ubyte");
  auto rawValidationLabels = loadLabels("../data/mnist/t10k-labels-idx1-ubyte");
  auto validationImages = Tensor<float>::stack(rawValidationImages.begin(),
                                               rawValidationImages.end());
  auto validationLabels = Tensor<float>::stack(rawValidationLabels.begin(),
                                               rawValidationLabels.end());

  // Define the model architecture
  LinearLayer layer0(784, 16);
  ReLU layer0Activation;
  LinearLayer layer1(16, 16);
  ReLU layer1Activation;
  LinearLayer layer2(16, 10);

  // Load weights
  layer0.weight = deserializeTensor<float>(file).value();
  layer0.bias = deserializeTensor<float>(file).value();
  layer1.weight = deserializeTensor<float>(file).value();
  layer1.bias = deserializeTensor<float>(file).value();
  layer2.weight = deserializeTensor<float>(file).value();
  layer2.bias = deserializeTensor<float>(file).value();

  file.close();

  // Inference
  auto z0 = layer0.forward(validationImages);
  auto a0 = layer0Activation.forward(z0);
  auto z1 = layer1.forward(a0);
  auto a1 = layer1Activation.forward(z1);
  auto z2 = layer2.forward(a1);
  auto result = z2.softmax(1);

  // Calculate accuracy
  auto predictedLabels = result.argmax(1);
  auto trueLabels = validationLabels.argmax(1);

  size_t correct = 0;
  for (size_t i = 0; i < predictedLabels.size; i++) {
    if (predictedLabels.data[i] == trueLabels.data[i]) {
      correct++;
    }
  }

  float accuracy = static_cast<float>(correct) / rawValidationImages.size();
  fmt::println("Validation Accuracy: {:.2f}%", accuracy * 100);

  return 0;
}
