#include "cppdl/graph.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

int main() {
  Graph g(1024);

  NodeId input =
      g.addNode<TensorNode>("input", std::vector<std::size_t>({30, 784}));

  NodeId weightsL0 =
      g.addNode<TensorNode>("weightsL0", std::vector<std::size_t>({784, 16}));
  NodeId biasL0 =
      g.addNode<TensorNode>("biasL0", std::vector<std::size_t>({16}));

  NodeId r0 = g.addNode<MatMulNode>(input, weightsL0,
                                    std::vector<std::size_t>({30, 16}));
  NodeId r1 =
      g.addNode<AddNode>(r0, biasL0, std::vector<std::size_t>({30, 16}));
  NodeId output = g.addNode<ReLUNode>(r1, std::vector<std::size_t>({30, 16}));

  printGraph(output);
}
