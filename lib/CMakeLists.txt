add_cppdl_library(cppdl mnist_utils.cpp) # iree_compile.cpp iree_run.cpp
#target_link_libraries(
#  cppdl
#  PRIVATE iree_compiler_API_SharedImpl StablehloOps iree_runtime_runtime
#          iree_hal_drivers_local_sync_sync_driver
#          iree_hal_local_loaders_embedded_elf_loader)
