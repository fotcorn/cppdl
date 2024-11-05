
#include "cppdl/codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace {
mlir::TensorType tensorTypeFromShape(std::vector<size_t> shape) {
  return mlir::TensorType::get(shape, mlir::Float32Type::get());
}
} // namespace

std::string codegen(const NeuralNetwork &nn) {
  // Initialize MLIR
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::stablehlo::StablehloDialect>();
  mlir::ImplicitLocOpBuilder b(mlir::UnknownLoc::get(&ctx), &ctx);
  mlir::ModuleOp module = b.create<mlir::ModuleOp>();

  // Linearize graph.
  auto nodes = nn.topologicalSort();
  std::unordered_map<NodeId, mlir::Value> code;

  auto func = b.create<mlir::func::FuncOp>(
      "main",
      b.getFunctionType({tensor28x28Type, tensor784x10Type, tensor1x10Type},
                        {tensor1x10Type}));

  // Generate MLIR module
  for (auto nodeId : nodes) {
    auto node = nn.getGraph().getNode(nodeId);
    switch (node->getKind()) {
    case NodeKind::MatMul: {
      MatMulNode *matMulNode = cast<MatMulNode>(node);

      auto inputA = code[matMulNode->getInputA()];
      auto inputB = code[matMulNode->getInputB()];

      auto dotOp = b.create<mlir::stablehlo::DotOp>(
          tensorTypeFromShape(matMulNode->getShape()), inputA, inputB,
          b.getI32ArrayAttr({}));

      code[nodeId] = dotOp.getResult();

      break;
    }
    }
  }
}