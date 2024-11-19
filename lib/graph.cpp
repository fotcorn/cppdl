#include "cppdl/graph.h"

void printNodeBackwards(const Graph &graph, NodeId nodeId) {
  Node *node = graph.getNode(nodeId);

  switch (node->getKind()) {
  case NodeKind::Add: {
    AddNode *addNode = cast<AddNode>(node);
    fmt::println("{} [label=\"Add {}\"];", nodeId, addNode->getShape());
    fmt::println("{} -> {};", addNode->getInputA(), nodeId);
    fmt::println("{} -> {};", addNode->getInputB(), nodeId);
    printNodeBackwards(graph, addNode->getInputA());
    printNodeBackwards(graph, addNode->getInputB());
    break;
  }
  case NodeKind::MatMul: {
    MatMulNode *matMulNode = cast<MatMulNode>(node);
    fmt::println("{} [label=\"MatMul {}\"];", nodeId, matMulNode->getShape());
    fmt::println("{} -> {};", matMulNode->getInputA(), nodeId);
    fmt::println("{} -> {};", matMulNode->getInputB(), nodeId);
    printNodeBackwards(graph, matMulNode->getInputA());
    printNodeBackwards(graph, matMulNode->getInputB());
    break;
  }
  case NodeKind::ReLU: {
    ReLUNode *reluNode = cast<ReLUNode>(node);
    fmt::println("{} [label=\"ReLU {}\"];", nodeId, reluNode->getShape());
    fmt::println("{} -> {};", reluNode->getInput(), nodeId);
    printNodeBackwards(graph, reluNode->getInput());
    break;
  }
  case NodeKind::Tensor: {
    TensorNode *tensorNode = cast<TensorNode>(node);
    fmt::println("{} [label=\"{} {}\", shape=box];", nodeId,
                 tensorNode->getName(), tensorNode->getShape());
    break;
  }
  case NodeKind::Softmax: {
    SoftmaxNode *softmaxNode = cast<SoftmaxNode>(node);
    fmt::println("{} [label=\"Softmax {}\"];", nodeId, softmaxNode->getShape());
    fmt::println("{} -> {};", softmaxNode->getInput(), nodeId);
    printNodeBackwards(graph, softmaxNode->getInput());
    break;
  }
  default:
    fmt::println("Unknown node kind");
    assert(false);
  }
}

void printGraphBackwards(const Graph &graph, NodeId outputNodeId) {
  fmt::println("digraph {{");
  printNodeBackwards(graph, outputNodeId);
  fmt::println("}}");
}
