add_cppdl_library(cppdl mnist_utils.cpp codegen.cpp graph.cpp)
target_include_directories(cppdl SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(cppdl PRIVATE MLIRIR MLIRParser MLIRSupport MLIRTransforms MLIRFuncDialect MLIRArithDialect MLIRLinalgDialect)
