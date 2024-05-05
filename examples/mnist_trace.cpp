#include "cppdl/trace.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

int main() {
  NeuralNetwork nn;

  auto w0 = nn.paramTensor("layer0.weight", {784, 16});
  auto b0 = nn.paramTensor("layer0.bias", {16});
  auto w1 = nn.paramTensor("layer1.weight", {16, 16});
  auto b1 = nn.paramTensor("layer1.bias", {16});
  auto w2 = nn.paramTensor("layer2.weight", {16, 10});
  auto b2 = nn.paramTensor("layer2.bias", {10});

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

  printGraphBackwards(nn.getGraph(), result.nodeId);

  return 0;
}
