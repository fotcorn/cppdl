if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_link_options("-fuse-ld=lld")
endif()

add_cppdl_executable(mnist ${CMAKE_CURRENT_SOURCE_DIR}/mnist.cpp)
add_cppdl_executable(mnist_inference
                     ${CMAKE_CURRENT_SOURCE_DIR}/mnist_inference.cpp)
add_cppdl_executable(graph ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp)
add_cppdl_executable(trace ${CMAKE_CURRENT_SOURCE_DIR}/trace.cpp)
add_cppdl_executable(mnist_trace ${CMAKE_CURRENT_SOURCE_DIR}/mnist_trace.cpp)

add_cppdl_executable(mlir_matmul ${CMAKE_CURRENT_SOURCE_DIR}/mlir_matmul.cpp)
target_include_directories(mlir_matmul SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS})
