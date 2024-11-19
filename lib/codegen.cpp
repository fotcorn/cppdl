
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
} // namespace

bool codegen(const NeuralNetwork &nn) {
  // Initialize MLIR
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  mlir::ImplicitLocOpBuilder b(mlir::UnknownLoc::get(&ctx), &ctx);
  mlir::ModuleOp module = b.create<mlir::ModuleOp>();

  auto nodes = nn.topologicalSort();

  // Function definition
  auto inputNodeId = nn.getInputTensor();
  auto inputNode = nn.getGraph().getNode(inputNodeId);
  auto outputNode = nn.getGraph().getNode(nodes.back());

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

  /*
  for (auto nodeId : nodes) {
    auto node = nn.getGraph().getNode(nodeId);
    switch (node->getKind()) {
    case NodeKind::MatMul: {
      MatMulNode *matMulNode = cast<MatMulNode>(node);

      auto inputA = code[matMulNode->getInputA()];
      auto inputB = code[matMulNode->getInputB()];

      auto dotOp = b.create<mlir::stablehlo::DotOp>(
          tensorTypeFromShape(matMulNode->getShape(), b), inputA, inputB,
          b.getI32ArrayAttr({}));

      code[nodeId] = dotOp.getResult();

      break;
    }
    case NodeKind::Add: {
      AddNode *addNode = cast<AddNode>(node);

      break;
    }
    case NodeKind::ReLU: {
      ReLUNode *reluNode = cast<ReLUNode>(node);

      break;
    }
    }
  }*/

  module.push_back(func);

  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return false;
  }

  module.print(llvm::outs());

  return true;
}
