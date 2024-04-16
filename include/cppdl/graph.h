#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

using NodeId = std::uintptr_t;
class Node;

class Graph {
  char *graphMemory;
  char *nextMemory;

public:
  Graph(std::size_t memory)
      : graphMemory(new char[memory]), nextMemory(graphMemory) {}
  ~Graph() { delete[] graphMemory; }

  template <typename GraphNodeType, typename... Args>
  NodeId addNode(Args &&...args) {
    static_assert(std::is_base_of<Node, GraphNodeType>::value,
                  "GraphNodeType must be a derivative of Node");
    new (nextMemory) GraphNodeType(std::forward<Args>(args)...);
    NodeId id = reinterpret_cast<NodeId>(nextMemory);
    nextMemory += sizeof(GraphNodeType);
    return id;
  }
};

enum class NodeKind {
  Add,
  Sub,
  Mul,
  Div,
  MatMul,
  Tensor,
  ReLU,
  Transpose,
  Reshape,
  Sum,
  Softmax,
};

class Node {
  NodeKind nodeKind;
  std::vector<std::size_t> shape;

protected:
  Node(NodeKind type, std::vector<std::size_t> shape)
      : nodeKind(type), shape(shape) {}

public:
  NodeKind getKind() const { return nodeKind; }
  std::vector<std::size_t> getShape() const { return shape; }
};

class BinaryNode : public Node {
  NodeId inputA;
  NodeId inputB;

public:
  NodeId getInputA() const { return inputA; }
  NodeId getInputB() const { return inputB; }

protected:
  BinaryNode(NodeKind type, std::vector<std::size_t> shape, NodeId inputA,
             NodeId inputB)
      : Node(type, shape), inputA(inputA), inputB(inputB) {}
};

class UnaryNode : public Node {
  NodeId input;

public:
  NodeId getInput() const { return input; }

protected:
  UnaryNode(NodeKind type, std::vector<std::size_t> shape, NodeId input)
      : Node(type, shape), input(input) {}
};

class AddNode : public BinaryNode {
public:
  AddNode(NodeId inputA, NodeId inputB, std::vector<std::size_t> shape)
      : BinaryNode(NodeKind::Add, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Add;
  }
};

class SubNode : public BinaryNode {
public:
  SubNode(NodeId inputA, NodeId inputB, std::vector<std::size_t> shape)
      : BinaryNode(NodeKind::Sub, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Sub;
  }
};

class MulNode : public BinaryNode {
public:
  MulNode(NodeId inputA, NodeId inputB, std::vector<std::size_t> shape)
      : BinaryNode(NodeKind::Mul, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Mul;
  }
};

class DivNode : public BinaryNode {
public:
  DivNode(NodeId inputA, NodeId inputB, std::vector<std::size_t> shape)
      : BinaryNode(NodeKind::Div, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Div;
  }
};

class MatMulNode : public BinaryNode {
public:
  MatMulNode(NodeId inputA, NodeId inputB, std::vector<std::size_t> shape)
      : BinaryNode(NodeKind::MatMul, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::MatMul;
  }
};

class TensorNode : public UnaryNode {
  std::string name;

public:
  TensorNode(std::string name, std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::Tensor, shape, std::numeric_limits<NodeId>::max()),
        name(std::move(name)) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Tensor;
  }
  std::string getName() const { return name; }
};

class ReLUNode : public UnaryNode {
public:
  ReLUNode(NodeId input, std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::ReLU, shape, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::ReLU;
  }
};
class TransposeNode : public UnaryNode {
public:
  TransposeNode(NodeId input, std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::Transpose, shape, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Transpose;
  }
};
class ReshapeNode : public UnaryNode {

public:
  ReshapeNode(NodeId input, std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::Reshape, shape, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Reshape;
  }
};
class SumNode : public UnaryNode {
  std::size_t dim;

public:
  SumNode(NodeId input, std::size_t dim, std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::Sum, shape, input), dim(dim) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Sum;
  }
  std::size_t getDim() const { return dim; }
};
class SoftmaxNode : public UnaryNode {
  std::size_t dim;

public:
  SoftmaxNode(NodeId input, std::size_t dim, std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::Softmax, shape, input), dim(dim) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Softmax;
  }
  std::size_t getDim() const { return dim; }
};

template <typename ToType, typename FromType>
ToType *cast(FromType *object) {
  assert(ToType::classof(object));
  return reinterpret_cast<ToType *>(object);
}

void printNode(NodeId nodeId) {
  Node *node = reinterpret_cast<Node *>(nodeId);

  switch (node->getKind()) {
  case NodeKind::Add: {
    AddNode *addNode = cast<AddNode>(node);
    fmt::println("{} [label=\"Add {}\"];", nodeId, addNode->getShape());
    fmt::println("{} -> {};", addNode->getInputA(), nodeId);
    fmt::println("{} -> {};", addNode->getInputB(), nodeId);
    printNode(addNode->getInputA());
    printNode(addNode->getInputB());
    break;
  }
  case NodeKind::MatMul: {
    MatMulNode *matMulNode = cast<MatMulNode>(node);
    fmt::println("{} [label=\"MatMul {}\"];", nodeId, matMulNode->getShape());
    fmt::println("{} -> {};", matMulNode->getInputA(), nodeId);
    fmt::println("{} -> {};", matMulNode->getInputB(), nodeId);
    printNode(matMulNode->getInputA());
    printNode(matMulNode->getInputB());
    break;
  }
  case NodeKind::ReLU: {
    ReLUNode *reluNode = cast<ReLUNode>(node);
    fmt::println("{} [label=\"ReLU {}\"];", nodeId, reluNode->getShape());
    fmt::println("{} -> {};", reluNode->getInput(), nodeId);
    printNode(reluNode->getInput());
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
    printNode(softmaxNode->getInput());
    break;
  }
  default:
    fmt::println("Unknown node kind");
    assert(false);
  }
}

void printGraph(NodeId outputNodeId) {
  fmt::println("digraph {{");
  printNode(outputNodeId);
  fmt::println("}}");
}
