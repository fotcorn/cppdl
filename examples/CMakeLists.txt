if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_link_options("-fuse-ld=lld")
endif()

add_cppdl_executable(mnist ${CMAKE_CURRENT_SOURCE_DIR}/mnist.cpp)
add_cppdl_executable(mnist_inference
                     ${CMAKE_CURRENT_SOURCE_DIR}/mnist_inference.cpp)
add_cppdl_executable(graph ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp)
add_cppdl_executable(trace ${CMAKE_CURRENT_SOURCE_DIR}/trace.cpp)
add_cppdl_executable(mnist_trace ${CMAKE_CURRENT_SOURCE_DIR}/mnist_trace.cpp)
if(CPPDL_USE_IREE)
  add_cppdl_executable(iree_compile
                       ${CMAKE_CURRENT_SOURCE_DIR}/iree_compile.cpp)
  target_link_libraries(iree_compile PRIVATE
    iree_compiler_API_StaticImpl
    StablehloOps
  )

  add_cppdl_executable(iree_run ${CMAKE_CURRENT_SOURCE_DIR}/iree_run.cpp)
  target_link_libraries(
    iree_run
    PRIVATE iree_compiler_bindings_c_loader iree_runtime_runtime
            iree_hal_drivers_local_sync_sync_driver
            iree_hal_local_loaders_embedded_elf_loader)

endif()
