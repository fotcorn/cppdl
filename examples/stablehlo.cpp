#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "stablehlo/dialect/StablehloOps.h"

int main() {
  mlir::MLIRContext ctx;

  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::stablehlo::StablehloDialect>();

  mlir::ImplicitLocOpBuilder b(mlir::UnknownLoc::get(&ctx), &ctx);

  mlir::ModuleOp module = b.create<mlir::ModuleOp>();

  auto tensor28x28Type = mlir::RankedTensorType::get({28, 28}, b.getF32Type());
  auto tensor784x10Type =
      mlir::RankedTensorType::get({784, 10}, b.getF32Type());
  auto tensor1x10Type = mlir::RankedTensorType::get({1, 10}, b.getF32Type());
  auto tensor1x784Type = mlir::RankedTensorType::get({1, 784}, b.getF32Type());

  auto func = b.create<mlir::func::FuncOp>(
      "main",
      b.getFunctionType({tensor28x28Type, tensor784x10Type, tensor1x10Type},
                        {tensor1x10Type}));

  auto entryBlock = func.addEntryBlock();
  b.setInsertionPointToStart(entryBlock);

  mlir::Value image = entryBlock->getArgument(0);
  mlir::Value weights = entryBlock->getArgument(1);
  mlir::Value bias = entryBlock->getArgument(2);

  mlir::Value reshapeOp =
      b.create<mlir::stablehlo::ReshapeOp>(tensor1x784Type, image);
  auto dotOp = b.create<mlir::stablehlo::DotOp>(tensor1x10Type, reshapeOp,
                                                weights, b.getI32ArrayAttr({}));
  auto addOp = b.create<mlir::stablehlo::AddOp>(tensor1x10Type, dotOp, bias);

  auto zero = b.create<mlir::stablehlo::ConstantOp>(
      mlir::DenseElementsAttr::get(tensor1x10Type, {0.0f}));

  auto maxOp = b.create<mlir::stablehlo::MaxOp>(tensor1x10Type, addOp, zero);
  b.create<mlir::func::ReturnOp>(maxOp.getResult());

  module.push_back(func);

  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "module verification failed\n";
  }

  module.dump();

  return 0;
}