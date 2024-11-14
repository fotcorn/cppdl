#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

void generateMLIRFunction(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

  mlir::OpBuilder b(&context);

  mlir::ModuleOp module = mlir::ModuleOp::create(b.getUnknownLoc());

  // Define the tensor types with specified dimensions
  mlir::Type inputType = mlir::RankedTensorType::get({1, 784}, b.getF32Type());
  mlir::Type weightType =
      mlir::RankedTensorType::get({784, 784}, b.getF32Type());
  mlir::Type biasType = mlir::RankedTensorType::get({1, 784}, b.getF32Type());
  auto resultType = mlir::RankedTensorType::get({1, 784}, b.getF32Type());

  // Function setup
  auto funcType =
      b.getFunctionType({inputType, weightType, biasType}, {resultType});

  auto func =
      b.create<mlir::func::FuncOp>(b.getUnknownLoc(), "matmul_add", funcType);
  mlir::Block *entryBlock = func.addEntryBlock();

  mlir::Value input = entryBlock->getArgument(0);
  mlir::Value weights = entryBlock->getArgument(1);
  mlir::Value bias = entryBlock->getArgument(2);

  b.setInsertionPointToStart(entryBlock);

  // Matmul of weights
  auto init = b.create<mlir::tensor::EmptyOp>(
      b.getUnknownLoc(), resultType.getShape(), b.getF32Type());

  auto matmul = b.create<mlir::linalg::MatmulOp>(
      b.getUnknownLoc(), mlir::ValueRange{input, weights},
      mlir::ValueRange{init});

  // Add bias
  auto result = b.create<mlir::arith::AddFOp>(b.getUnknownLoc(),
                                              matmul.getResult(0), bias);

  b.create<mlir::func::ReturnOp>(b.getUnknownLoc(), result.getResult());

  module.push_back(func);

  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return;
  }

  module.print(llvm::outs());
}

int main() {
  mlir::MLIRContext context;

  generateMLIRFunction(context);

  return 0;
}
