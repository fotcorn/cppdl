#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "cppdl/graph.h"

class NeuralNetwork;

class TraceTensor {
  Graph &graph;

public:
  NodeId nodeId;
  std::vector<std::size_t> shape;

  TraceTensor(std::vector<std::size_t> shape, Graph &graph, NodeId nodeId)
      : graph(graph), nodeId(nodeId), shape(shape) {}

  TraceTensor matmul(TraceTensor &other) const {
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

  TraceTensor add(TraceTensor &other) const {
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

  TraceTensor softmax(size_t dim) {
    NodeId output = graph.addNode<SoftmaxNode>(nodeId, dim,
                                               std::vector<std::size_t>(shape));
    return TraceTensor(shape, graph, output);
  }

  TraceTensor sum(size_t dim) {
    NodeId output =
        graph.addNode<SumNode>(nodeId, dim, std::vector<std::size_t>(shape));
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

  TraceTensor paramTensor(std::string name, std::vector<std::size_t> shape) {
    auto id = graph.addNode<TensorNode>(name, std::vector<std::size_t>(shape));
    paramTensors.push_back(id);
    return TraceTensor(shape, graph, id);
  }

  TraceTensor inputTensor(std::string name, std::vector<std::size_t> shape) {
    auto id = graph.addNode<TensorNode>(name, std::vector<std::size_t>(shape));
    inputTensors.push_back(id);
    return TraceTensor(shape, graph, id);
  }

  const Graph &getGraph() { return graph; }

  std::vector<NodeId> topologicalSort() {
    // TODO: cycle detection.
    std::vector<NodeId> list;

    std::function<void(NodeId)> topoVisit = [&](NodeId nodeId) {
      if (std::find(list.begin(), list.end(), nodeId) != list.end()) {
        return;
      }
      Node *node = graph.getNode(nodeId);
      for (NodeId successor : node->getSuccessors()) {
        topoVisit(successor);
      }
      list.push_back(nodeId);
    };

    for (auto nodeId : inputTensors) {
      topoVisit(nodeId);
    }
    for (auto nodeId : paramTensors) {
      topoVisit(nodeId);
    }

    std::reverse(list.begin(), list.end());
    return list;
  }
};

class LinearLayer {
  TraceTensor weight;
  TraceTensor bias;

public:
  LinearLayer(TraceTensor weight, TraceTensor bias)
      : weight(weight), bias(bias) {}

  TraceTensor forward(const TraceTensor &input) {
    auto res1 = input.matmul(weight);
    auto res2 = res1.add(bias);
    return res2;
  }
};