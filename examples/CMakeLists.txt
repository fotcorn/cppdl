add_cppdl_executable(mnist ${CMAKE_CURRENT_SOURCE_DIR}/mnist.cpp)
add_cppdl_executable(mnist_inference
                     ${CMAKE_CURRENT_SOURCE_DIR}/mnist_inference.cpp)
add_cppdl_executable(codegen ${CMAKE_CURRENT_SOURCE_DIR}/codegen.cpp)
add_cppdl_executable(graph ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp)
if(CPPDL_USE_IREE)
  add_cppdl_executable(iree_compile
                       ${CMAKE_CURRENT_SOURCE_DIR}/iree_compile.cpp)
  target_link_libraries(iree_compile PRIVATE iree_compiler_bindings_c_loader)
endif()
