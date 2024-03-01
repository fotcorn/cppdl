// clang-format off

/*
func.func @main(
  %alpha: tensor<f32>, %x: tensor<4xf32>, %y: tensor<4xf32>
) -> tensor<4xf32> {
  %0 = stablehlo.broadcast_in_dim %alpha, dims = []
    : (tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.multiply %0, %x : tensor<4xf32>
  %2 = stablehlo.add %1, %y : tensor<4xf32>
  func.return %2: tensor<4xf32>
}

module {
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1)):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  }) {sym_name = "multiply_transpose", type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
*/

// clang-format on

#include <string>

namespace mm {

class Module {
public:
  TypeBuilder type(std::string name) { return TypeBuilder(name); }
  OpBuilder op(std::string operation) { return OpBuilder(operation); }
};

class Type {};
class TypeBuilder {
  TypeBuilder(std::string name) {}

public:
  TypeBuilder &param(const Type &type) { return *this; }
  TypeBuilder &param(std::string) { return *this; }
  Type build(){};

  friend class Module;
};

class OpBuilder {
  OpBuilder(std::string operation) {}

  friend class Module;
};

}; // namespace mm

int main() {

  mm::Module m;

  auto f32 = m.type("f32").build();
  auto tensorF32 = m.type("tensor").param(f32).build();
  auto t4xf32 = m.type("tensor").param("4xf32").build();

  auto ret = m.functionType().param(f32).returnType(f32).build();

  auto funcOp = m.op("func.func")
                    .attr("sym_name", "main")
                    .attr("type", ret)
                    .param("alpha", tensorF32)
                    .type(f32)
                    .build();

  auto block = funcOp.region().block("bb0");
  auto add1 = block.op("arith.add")
                  .type(f32)
                  .param(funcOp.result(0))
                  .param(funcOp.result(1))
                  .build();
  auto add2 = block.op("arith.add")
                  .type(f32)
                  .param(funcOp.result(0))
                  .param(add1.result(1))
                  .build();
}
