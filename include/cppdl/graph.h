#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "tensor.h"

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
    new (nextMemory) GraphNodeType(*this, std::forward<Args>(args)...);
    NodeId id = reinterpret_cast<NodeId>(nextMemory);
    nextMemory += sizeof(GraphNodeType);
    return id;
  }

  Node *getNode(NodeId id) const { return reinterpret_cast<Node *>(id); }
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
  std::vector<NodeId> successors;

protected:
  Node(NodeKind type, std::vector<std::size_t> shape)
      : nodeKind(type), shape(shape) {}

public:
  NodeKind getKind() const { return nodeKind; }
  std::vector<std::size_t> getShape() const { return shape; }
  void addSuccessor(NodeId successor) { successors.push_back(successor); }
  const std::vector<NodeId> &getSuccessors() const { return successors; }
};

class BinaryNode : public Node {
  NodeId inputA;
  NodeId inputB;

public:
  NodeId getInputA() const { return inputA; }
  NodeId getInputB() const { return inputB; }

protected:
  BinaryNode(Graph &graph, NodeKind type, std::vector<std::size_t> shape,
             NodeId inputA, NodeId inputB)
      : Node(type, shape), inputA(inputA), inputB(inputB) {
    graph.getNode(inputA)->addSuccessor(reinterpret_cast<NodeId>(this));
    graph.getNode(inputB)->addSuccessor(reinterpret_cast<NodeId>(this));
  }
};

class UnaryNode : public Node {
  NodeId input;

public:
  NodeId getInput() const { return input; }

protected:
  UnaryNode(Graph &graph, NodeKind type, std::vector<std::size_t> shape,
            NodeId input)
      : Node(type, shape), input(input) {
    graph.getNode(input)->addSuccessor(reinterpret_cast<NodeId>(this));
  }
};

class AddNode : public BinaryNode {
public:
  AddNode(Graph &graph, NodeId inputA, NodeId inputB,
          std::vector<std::size_t> shape)
      : BinaryNode(graph, NodeKind::Add, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Add;
  }
};

class SubNode : public BinaryNode {
public:
  SubNode(Graph &graph, NodeId inputA, NodeId inputB,
          std::vector<std::size_t> shape)
      : BinaryNode(graph, NodeKind::Sub, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Sub;
  }
};

class MulNode : public BinaryNode {
public:
  MulNode(Graph &graph, NodeId inputA, NodeId inputB,
          std::vector<std::size_t> shape)
      : BinaryNode(graph, NodeKind::Mul, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Mul;
  }
};

class DivNode : public BinaryNode {
public:
  DivNode(Graph &graph, NodeId inputA, NodeId inputB,
          std::vector<std::size_t> shape)
      : BinaryNode(graph, NodeKind::Div, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Div;
  }
};

class MatMulNode : public BinaryNode {
public:
  MatMulNode(Graph &graph, NodeId inputA, NodeId inputB,
             std::vector<std::size_t> shape)
      : BinaryNode(graph, NodeKind::MatMul, shape, inputA, inputB) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::MatMul;
  }
};

class TensorNode : public Node {
  std::string name;
  Tensor<float> data;

public:
  TensorNode(Graph &, std::string name, const Tensor<float>& data = Tensor<float>())
      : Node(NodeKind::Tensor, data.shape), name(name), data(data) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Tensor;
  }
  std::string getName() const { return name; }
  const Tensor<float>& getData() const { return data; }
};

class ReLUNode : public UnaryNode {
public:
  ReLUNode(Graph &graph, NodeId input, std::vector<std::size_t> shape)
      : UnaryNode(graph, NodeKind::ReLU, shape, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::ReLU;
  }
};
class TransposeNode : public UnaryNode {
public:
  TransposeNode(Graph &graph, NodeId input, std::vector<std::size_t> shape)
      : UnaryNode(graph, NodeKind::Transpose, shape, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Transpose;
  }
};
class ReshapeNode : public UnaryNode {

public:
  ReshapeNode(Graph &graph, NodeId input, std::vector<std::size_t> shape)
      : UnaryNode(graph, NodeKind::Reshape, shape, input) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Reshape;
  }
};
class SumNode : public UnaryNode {
  std::size_t dim;

public:
  SumNode(Graph &graph, NodeId input, std::size_t dim,
          std::vector<std::size_t> shape)
      : UnaryNode(graph, NodeKind::Sum, shape, input), dim(dim) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Sum;
  }
  std::size_t getDim() const { return dim; }
};
class SoftmaxNode : public UnaryNode {
  std::size_t dim;

public:
  SoftmaxNode(Graph &graph, NodeId input, std::size_t dim,
              std::vector<std::size_t> shape)
      : UnaryNode(graph, NodeKind::Softmax, shape, input), dim(dim) {}
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

void printNodeBackwards(const Graph &graph, NodeId nodeId);
void printGraphBackwards(const Graph &graph, NodeId outputNodeId);
