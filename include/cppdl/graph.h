#pragma once

#include <cstdint>
#include <limits>
#include <vector>

// https://gist.github.com/Leedehai/535a73e19c1cdb2b684279d377d2da52

// TODO
// manual rtti

// Idea: decrease size of used memory of nodes. Index into graphMemory.
using NodeId = uint16_t;
class Node;
class Graph {
  std::vector<uint8_t> graphMemory;

public:
  template <typename GraphNodeType, typename... Args>
  NodeId addNode(Args &&...args) {
    static_assert(std::is_base_of<Node, GraphNodeType>::value,
                  "GraphNodeType must be a derivative of Node");
    new GraphNodeType(std::forward<Args>(args)...);
    return 0;
  }
};

enum class NodeKind {
  Add,
  Sub,
  Mul,
  Div,
  MatMul,
  Input,
  ReLU,
  Transpose,
  Reshape,
  Sum,
};

class Node {
  std::vector<std::size_t> shape;
  NodeKind nodeKind;

protected:
  Node(NodeKind type) : nodeKind(type) {}

public:
  NodeKind getKind() const { return nodeKind; }
};

class BinaryNode : public Node {
  NodeId inputA;
  NodeId inputB;

public:
  NodeId getInputA() const { return inputA; }
  NodeId getInputB() const { return inputB; }

protected:
  BinaryNode(NodeKind type, NodeId inputA, NodeId inputB)
      : Node(type), inputA(inputA), inputB(inputB) {}
};

class UnaryNode : public Node {
  NodeId input;

public:
  NodeId getInput() const { return input; }

protected:
  UnaryNode(NodeKind type, NodeId input) : Node(type), input(input) {}
};

class AddNode : public BinaryNode {
public:
  AddNode(NodeId inputA, NodeId inputB)
      : BinaryNode(NodeKind::Add, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Add;
  }
};

class SubNode : public BinaryNode {
public:
  SubNode(NodeId inputA, NodeId inputB)
      : BinaryNode(NodeKind::Sub, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Sub;
  }
};

class MulNode : public BinaryNode {
public:
  MulNode(NodeId inputA, NodeId inputB)
      : BinaryNode(NodeKind::Mul, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Mul;
  }
};

class DivNode : public BinaryNode {
public:
  DivNode(NodeId inputA, NodeId inputB)
      : BinaryNode(NodeKind::Div, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Div;
  }
};

class MatMulNode : public BinaryNode {
public:
  MatMulNode(NodeId inputA, NodeId inputB)
      : BinaryNode(NodeKind::MatMul, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::MatMul;
  }
};

class InputNode : UnaryNode {
public:
  InputNode(std::vector<std::size_t>)
      : UnaryNode(NodeKind::Input, std::numeric_limits<NodeId>::max()) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Input;
  }
};
class ReLUNode : UnaryNode {
public:
  ReLUNode(NodeId input) : UnaryNode(NodeKind::ReLU, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::ReLU;
  }
};
class TransposeNode : UnaryNode {
public:
  TransposeNode(NodeId input) : UnaryNode(NodeKind::Transpose, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Transpose;
  }
};
class ReshapeNode : UnaryNode {
  std::vector<std::size_t> newShape;

public:
  ReshapeNode(NodeId input, std::vector<std::size_t> newShape)
      : UnaryNode(NodeKind::Reshape, input), newShape(newShape) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Reshape;
  }
};
class SumNode : UnaryNode {
  std::size_t dim;

public:
  SumNode(NodeId input, std::size_t dim)
      : UnaryNode(NodeKind::Sum, input), dim(dim) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Sum;
  }
  std::size_t getDim() const { return dim; }
};

template <typename ToType, typename FromType>
ToType *cast(FromType *object) {
  assert(ToType::classof(object));
  return reinterpret_cast<ToType *>(object);
}
