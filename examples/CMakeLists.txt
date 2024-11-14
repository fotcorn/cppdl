if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_link_options("-fuse-ld=lld")
endif()

add_cppdl_executable(mnist ${CMAKE_CURRENT_SOURCE_DIR}/mnist.cpp)
add_cppdl_executable(mnist_inference
                     ${CMAKE_CURRENT_SOURCE_DIR}/mnist_inference.cpp)
add_cppdl_executable(graph ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp)
add_cppdl_executable(trace ${CMAKE_CURRENT_SOURCE_DIR}/trace.cpp)
add_cppdl_executable(mnist_trace ${CMAKE_CURRENT_SOURCE_DIR}/mnist_trace.cpp)

# if(CPPDL_USE_IREE)
#   add_cppdl_executable(iree_example
#                        ${CMAKE_CURRENT_SOURCE_DIR}/iree_example.cpp)
#   target_link_libraries(
#     iree_example
#     PRIVATE iree_compiler_API_StaticImpl StablehloOps iree_runtime_runtime
#             iree_hal_drivers_local_sync_sync_driver
#             iree_hal_local_loaders_embedded_elf_loader)
# 
add_cppdl_executable(stablehlo ${CMAKE_CURRENT_SOURCE_DIR}/stablehlo.cpp)
target_link_libraries(stablehlo PRIVATE MLIRIR MLIRParser MLIRSupport MLIRTransforms MLIRFuncDialect MLIRArithDialect MLIRLinalgDialect)
target_include_directories(stablehlo SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS})


# endif()
