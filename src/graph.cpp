#include "cppdl/graph.h"

int main() {
  Graph g;

  NodeId input = g.addNode<InputNode>(std::vector<std::size_t>({30, 784}));

  NodeId weightsL0 = g.addNode<InputNode>(std::vector<std::size_t>({784, 16}));
  NodeId biasL0 = g.addNode<InputNode>(std::vector<std::size_t>({16}));

  // NodeId r0 = g.addNode<MatMulNode>(input, weightsL0);
  // NodeId r1 = g.addNode<AddNode>(r0, biasL0);
  // NodeId r2 = g.addNode<ReLUNode>(r1);

  /*Node *node;
  switch (node->getKind()) {
  case NodeKind::Add:
    AddNode *addNode = cast<AddNode>(node);
    addNode->getInputA();
  }
  */
}
