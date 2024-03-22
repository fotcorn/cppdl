#include "cppdl/nn.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {

  // Define the model architecture
  LinearLayer layer0(784, 16);
  ReLU layer0Activation;
  LinearLayer layer1(16, 16);
  ReLU layer1Activation;
  LinearLayer layer2(16, 10);

  Graph graph;
  TraceContext ctx(graph);

  // Load weights
  layer0.weight = ctx.weightTensor({16, 784});
  layer0.bias = ctx.weightTensor({16});
  layer1.weight = ctx.weightTensor({16, 16});
  layer1.bias = ctx.weightTensor({16});
  layer2.weight = ctx.weightTensor({16, 10});
  layer2.bias = ctx.weightTensor({10});

  auto validationImages = ctx.inputTensor({25, 784});

  // Inference
  auto z0 = layer0.forward(validationImages);
  auto a0 = layer0Activation.forward(z0);
  auto z1 = layer1.forward(a0);
  auto a1 = layer1Activation.forward(z1);
  auto z2 = layer2.forward(a1);
  auto result = z2.softmax(1);

  ctx.output(result);

  graph.plot();

  return 0;
}
