#include "cppdl/iree.h"

int main() {
  auto compiler = iree::Compiler::create();
  if (!compiler) {
    return 1;
  }

  iree::CompilerSession session;

  std::string simple_mul_mlir = " \
func.func @simple_mul(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {\n\
  %result = arith.mulf %lhs, %rhs : tensor<4xf32>\n \
  return %result : tensor<4xf32>\n \
}";

  auto sourceWrapBuffer =
      session.createSourceWrapBuffer(simple_mul_mlir, "simple_mul");

  if (!sourceWrapBuffer) {
    return 1;
  }

  auto invocation = session.createCompilerInvocation();

  if (!invocation->parseSource(sourceWrapBuffer.get())) {
    fmt::println(stderr, "Error parsing source.");
    return 1;
  }

  if (!invocation->compile()) {
    fmt::println(stderr, "Error compiling source.");
    return 1;
  }

  invocation->outputIR();

  invocation->run();

  return 0;
}
