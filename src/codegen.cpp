#include "cppdl/tinymlir.h"

int main() {
  tmlir::Module m;

  auto f32 = m.type("f32").build();
  auto tensorF32 = m.type("tensor").param(f32).build();
  auto t4xf32 = m.type("tensor").param("4xf32").build();

  auto ret = m.functionType().param(f32).returnType(f32).build();

  auto funcOp = m.op("func.func")
                    .attr("sym_name", "main")
                    .attr("type", ret)
                    .param(tensorF32)
                    .type(f32)
                    .build();

  auto block = funcOp.region().block("bb0").param("arg0", f32).build();
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
