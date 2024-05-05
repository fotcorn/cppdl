#include "cppdl/trace.h"

int main() {
  NeuralNetwork nn;
  auto w1 = nn.paramTensor("w1", {16, 16});
  auto b1 = nn.paramTensor("b1", {16});
  auto w2 = nn.paramTensor("w2", {16, 16});
  auto b2 = nn.paramTensor("b2", {16});

  auto x = nn.setInputTensor("x", {10, 16});

  auto h1 = x.matmul(w1).add(b1).relu();
  auto y = h1.matmul(w2).add(b2).relu();

  printGraphBackwards(nn.getGraph(), y.nodeId);
}
