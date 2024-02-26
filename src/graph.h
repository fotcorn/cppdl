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

// TODO: constructor + classof
class InputNode : UnaryNode {
public:
  InputNode(std::vector<std::size_t> shape)
      : UnaryNode(NodeKind::Input, std::numeric_limits<NodeId>::max()) {}
  static bool classof(const Node *node) {
    return node->getKind() == NodeKind::Input;
  }
};
class ReLUNode : UnaryNode {
  ReLUNode(NodeId input) : UnaryNode(NodeKind::ReLU, input) {}
};
class TransposeNode : UnaryNode {};
class ReshapeNode : UnaryNode {
  std::vector<std::size_t> newShape;
};
class SumNode : UnaryNode {
  // std::size_t dim;
};

template <typename ToType, typename FromType>
ToType *cast(FromType *object) {
  assert(ToType::classof(object));
  return reinterpret_cast<ToType *>(object);
}

void func() {

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
