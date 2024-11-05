#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

void generateMLIRFunction(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

  mlir::OpBuilder builder(&context);

  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Define the tensor types with specified dimensions
  mlir::Type inputType =
      mlir::RankedTensorType::get({-1, 28}, builder.getF32Type());
  mlir::Type weightType =
      mlir::RankedTensorType::get({28, 28}, builder.getF32Type());
  mlir::Type biasType =
      mlir::RankedTensorType::get({1, 28}, builder.getF32Type());
  mlir::Type resultType =
      mlir::RankedTensorType::get({-1, 28}, builder.getF32Type());

  mlir::FunctionType funcType =
      builder.getFunctionType({inputType, weightType, biasType}, {resultType});

  auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                 "matmul_add", funcType);
  mlir::Block *entryBlock = func.addEntryBlock();

  mlir::Value input = entryBlock->getArgument(0);
  mlir::Value weights = entryBlock->getArgument(1);
  mlir::Value bias = entryBlock->getArgument(2);

  builder.setInsertionPointToStart(entryBlock);

  // Create the matrix multiplication operation
  auto matmul = builder.create<mlir::linalg::MatmulOp>(
      builder.getUnknownLoc(), input, weights, resultType);

  // Broadcast bias to match the result dimensions and add it element-wise
  auto broadcastBias = builder.create<mlir::linalg::BroadcastOp>(
      builder.getUnknownLoc(), resultType, bias, resultType);
  auto result = builder.create<mlir::arith::AddFOp>(
      builder.getUnknownLoc(), matmul.getResult(0), broadcastBias.getResult(0));

  // Return the result
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), result);

  module.push_back(func);

  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return;
  }

  module.print(llvm::outs());
}

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  generateMLIRFunction(context);

  return 0;
}
