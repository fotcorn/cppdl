add_executable(cppdl_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_elementwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_stack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nn_activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nn_linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_softmax.cpp
)

target_include_directories(cppdl_test PRIVATE ../src)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
target_compile_options(cppdl_test PRIVATE -Wall -Wextra -Wpedantic -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
target_compile_options(cppdl_test PRIVATE /W4)
endif()
target_compile_features(cppdl_test PUBLIC cxx_std_17)
target_link_libraries(cppdl_test gtest gtest_main fmt::fmt)

add_test(NAME cppdl_test COMMAND cppdl_test)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS cppdl_test)
