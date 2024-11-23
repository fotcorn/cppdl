
#include "cppdl/codegen.h"
#include "cppdl/graph.h"
#include "cppdl/trace.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

namespace {
mlir::RankedTensorType tensorTypeFromShape(std::vector<size_t> shape,
                                           mlir::OpBuilder &b) {
  std::vector<int64_t> shapeSigned;
  shapeSigned.reserve(shape.size());
  std::transform(shape.begin(), shape.end(), std::back_inserter(shapeSigned),
                 [](size_t dim) { return static_cast<int64_t>(dim); });
  return mlir::RankedTensorType::get(shapeSigned, b.getF32Type());
}

mlir::Value createEmptyTensor(Node* node, mlir::OpBuilder &b) {
      auto unsignedShape = node->getShape();
      auto shape = llvm::SmallVector<int64_t>(unsignedShape.begin(),
                                              unsignedShape.end());
      return b.create<mlir::tensor::EmptyOp>(b.getUnknownLoc(), shape,
                                                  b.getF32Type());
}
} // namespace

bool codegen(const NeuralNetwork &nn) {
  // Initialize MLIR
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::linalg::LinalgDialect>();
  mlir::ImplicitLocOpBuilder b(mlir::UnknownLoc::get(&ctx), &ctx);
  mlir::ModuleOp module = b.create<mlir::ModuleOp>();

  auto nodes = nn.topologicalSort();

  // Function definition
  auto inputNodeId = nn.getInputTensor();
  auto inputNode = nn.getGraph().getNode(inputNodeId);
  auto outputNodeId = nodes.back();
  auto outputNode = nn.getGraph().getNode(outputNodeId);

  auto inputShape = tensorTypeFromShape(inputNode->getShape(), b);
  auto outputShape = tensorTypeFromShape(outputNode->getShape(), b);

  auto funcType = b.getFunctionType({inputShape}, {outputShape});

  auto func =
      b.create<mlir::func::FuncOp>(b.getUnknownLoc(), "forward", funcType);
  mlir::Block *entryBlock = func.addEntryBlock();
  b.setInsertionPointToStart(entryBlock);

  mlir::Value input = entryBlock->getArgument(0);

  // Generate code for each node
  std::unordered_map<NodeId, mlir::Value> code;
  code[inputNodeId] = input;

  for (auto paramTensorId : nn.getParamTensors()) {
    auto paramTensor = nn.getGraph().getNode(paramTensorId);
    auto tensorNode = cast<TensorNode>(paramTensor);
    auto tensorType = tensorTypeFromShape(paramTensor->getShape(), b);
    auto tensorData = tensorNode->getData().data;
    auto binaryData =
        llvm::ArrayRef<char>(reinterpret_cast<char *>(tensorData.get()),
                             tensorNode->getData().size * sizeof(float));
    auto blob = mlir::UnmanagedAsmResourceBlob::allocateInferAlign(binaryData);
    auto denseAttr = mlir::DenseResourceElementsAttr::get(
        tensorType, tensorNode->getName(), std::move(blob));
    auto constTensor =
        b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), denseAttr);
    code[paramTensorId] = constTensor;
  }

  for (auto nodeId : nodes) {
    if (code.find(nodeId) != code.end()) {
      continue;
    }
    auto node = nn.getGraph().getNode(nodeId);
    switch (node->getKind()) {
    case NodeKind::MatMul: {
      MatMulNode *matMulNode = cast<MatMulNode>(node);

      auto inputA = code[matMulNode->getInputA()];
      auto inputB = code[matMulNode->getInputB()];

      auto init = createEmptyTensor(matMulNode, b);

      auto matmul = b.create<mlir::linalg::MatmulOp>(
          b.getUnknownLoc(), mlir::ValueRange{inputA, inputB},
          mlir::ValueRange{init});

      code[nodeId] = matmul.getResult(0);

      break;
    }
    case NodeKind::Add: {
      AddNode *addNode = cast<AddNode>(node);
      auto inputA = code[addNode->getInputA()];
      auto inputB = code[addNode->getInputB()];

      auto inputAShape = nn.getGraph().getNode(addNode->getInputA())->getShape();
      auto inputBShape = nn.getGraph().getNode(addNode->getInputB())->getShape();

      if (inputAShape.size() > inputBShape.size()) {
        auto init = createEmptyTensor(addNode, b);
        inputB = b.create<mlir::linalg::BroadcastOp>(b.getUnknownLoc(), inputB, init, 0).getResult()[0];
      } else if (inputBShape.size() > inputAShape.size()) {
        auto init = createEmptyTensor(addNode, b);
        inputA = b.create<mlir::linalg::BroadcastOp>(b.getUnknownLoc(), inputA, init, 0).getResult()[0];
      }
      auto result =
          b.create<mlir::arith::AddFOp>(b.getUnknownLoc(), inputA, inputB);
      code[nodeId] = result;
      break;
    }
    case NodeKind::ReLU: {
      ReLUNode *reluNode = cast<ReLUNode>(node);
      auto input = code[reluNode->getInput()];


      auto init = createEmptyTensor(reluNode, b);
      auto zero = b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getF32Type(), b.getF32FloatAttr(0.0));
      auto zeroTensor = b.create<mlir::linalg::FillOp>(b.getUnknownLoc(), mlir::ValueRange{zero}, mlir::ValueRange{init}).getResult(0);

      auto result =
          b.create<mlir::arith::MaximumFOp>(b.getUnknownLoc(), input, zeroTensor);
      code[nodeId] = result;
      break;
    }
    case NodeKind::Softmax: {
      SoftmaxNode *softmaxNode = cast<SoftmaxNode>(node);
      auto input = code[softmaxNode->getInput()];

      auto init = createEmptyTensor(softmaxNode, b);

      auto softmax =
          b.create<mlir::linalg::SoftmaxOp>(b.getUnknownLoc(), tensorTypeFromShape(softmaxNode->getShape(), b), input, init, 0);

      code[nodeId] = softmax.getResult()[0];
      break;
    }
    }
  }

  auto output = code[outputNodeId];
  b.create<mlir::func::ReturnOp>(b.getUnknownLoc(), mlir::ValueRange{output});

  module.push_back(func);

  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return false;
  }

  module.print(llvm::outs());
  return true;
}
