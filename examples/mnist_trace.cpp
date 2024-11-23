#include "cppdl/trace.h"
#include "cppdl/codegen.h"
#include "cppdl/serialization.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

int main(int argc, char* argv[]) {
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

  NeuralNetwork nn;

  // Load weights from the file
  auto w0_tensor = deserializeTensor<float>(file).value().transpose();
  auto b0_tensor = deserializeTensor<float>(file).value();
  auto w1_tensor = deserializeTensor<float>(file).value().transpose();
  auto b1_tensor = deserializeTensor<float>(file).value();
  auto w2_tensor = deserializeTensor<float>(file).value().transpose();
  auto b2_tensor = deserializeTensor<float>(file).value();

  // Initialize paramTensors with the loaded weights
  auto w0 = nn.paramTensor("layer0.weight", w0_tensor.shape, w0_tensor);
  auto b0 = nn.paramTensor("layer0.bias", b0_tensor.shape, b0_tensor);
  auto w1 = nn.paramTensor("layer1.weight", w1_tensor.shape, w1_tensor);
  auto b1 = nn.paramTensor("layer1.bias", b1_tensor.shape, b1_tensor);
  auto w2 = nn.paramTensor("layer2.weight", w2_tensor.shape, w2_tensor);
  auto b2 = nn.paramTensor("layer2.bias", b2_tensor.shape, b2_tensor);

  // Define the model architecture
  LinearLayer layer0(w0, b0);
  LinearLayer layer1(w1, b1);
  LinearLayer layer2(w2, b2);

  auto validationImages = nn.setInputTensor("image", {25, 784});

  // Inference
  auto z0 = layer0.forward(validationImages);
  auto a0 = z0.relu();
  auto z1 = layer1.forward(a0);
  auto a1 = z1.relu();
  auto z2 = layer2.forward(a1);
  auto result = z2.softmax(1);

  //printGraphBackwards(nn.getGraph(), result.nodeId);

  codegen(nn);

  return 0;
}
