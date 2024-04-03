#include <iostream>
#include <vector>

#include "cppdl/graph.h"
#include <fmt/core.h>
#include <fmt/ranges.h>

class NeuralNetwork;

class TraceTensor {
  Graph &graph;

public:
  NodeId nodeId;
  std::vector<std::size_t> shape;

  TraceTensor(std::vector<std::size_t> shape, Graph &graph, NodeId nodeId)
      : graph(graph), nodeId(nodeId), shape(shape) {}

  TraceTensor matmul(TraceTensor &other) {
    if (shape.size() != 2 || other.shape.size() != 2) {
      throw std::runtime_error(fmt::format(
          "matmul: both tensors must be 2-dimensional, got shapes {} and {}",
          shape, other.shape));
    }
    if (shape[1] != other.shape[0]) {
      throw std::runtime_error(
          "matmul: incompatible shapes for matrix multiplication");
    }
    std::vector<std::size_t> matmulShape = {shape[0], other.shape[1]};
    NodeId output =
        graph.addNode<MatMulNode>(nodeId, other.nodeId, matmulShape);
    return TraceTensor(matmulShape, graph, output);
  }

  TraceTensor add(TraceTensor &other) {
    auto &op1 = *this;
    auto &op2 = other;

    size_t length = std::max(op1.shape.size(), op2.shape.size());

    auto shapeOp1 = std::vector<size_t>(length, 1);
    auto shapeOp2 = std::vector<size_t>(length, 1);
    std::copy_backward(op1.shape.begin(), op1.shape.end(), shapeOp1.end());
    std::copy_backward(op2.shape.begin(), op2.shape.end(), shapeOp2.end());

    for (size_t i = 0; i < shapeOp1.size(); i++) {
      if (shapeOp1[i] != shapeOp2[i] && shapeOp1[i] != 1 && shapeOp2[i] != 1) {
        throw std::runtime_error(
            fmt::format("incompatible shapes for arithmetic operation "
                        "(broadcasting applied): {}, {}",
                        shapeOp1, shapeOp2));
      }
    }

    std::vector<std::size_t> addShape;
    for (size_t i = 0; i < length; i++) {
      addShape.push_back(std::max(shapeOp1[i], shapeOp2[i]));
    }

    NodeId output = graph.addNode<AddNode>(nodeId, other.nodeId,
                                           std::vector<std::size_t>(addShape));
    return TraceTensor(addShape, graph, output);
  }

  TraceTensor relu() {
    NodeId output =
        graph.addNode<ReLUNode>(nodeId, std::vector<std::size_t>(shape));
    return TraceTensor(shape, graph, output);
  }
};

class NeuralNetwork {
  std::vector<NodeId> paramTensors;
  std::vector<NodeId> inputTensors;

  Graph graph;

  friend class TraceTensor;

public:
  NeuralNetwork() : graph(1024 * 100) {}

  TraceTensor paramTensor(std::vector<std::size_t> shape) {
    auto id = graph.addNode<TensorNode>(std::vector<std::size_t>(shape));
    paramTensors.push_back(id);
    return TraceTensor(shape, graph, id);
  }

  TraceTensor inputTensor(std::vector<std::size_t> shape) {
    auto id = graph.addNode<TensorNode>(std::vector<std::size_t>(shape));
    inputTensors.push_back(id);
    return TraceTensor(shape, graph, id);
  }
};

int main() {
  NeuralNetwork nn;
  auto w1 = nn.paramTensor({16, 16});
  auto b1 = nn.paramTensor({16});
  auto w2 = nn.paramTensor({16, 16});
  auto b2 = nn.paramTensor({16});

  auto x = nn.inputTensor({10, 16});

  auto h1 = x.matmul(w1).add(b1).relu();
  auto y = h1.matmul(w2).add(b2).relu();

  printGraph(y.nodeId);
}
