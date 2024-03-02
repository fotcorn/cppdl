add_cppdl_executable(cppdl_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_elementwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_stack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_mse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nn_activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nn_linear.cpp
)

target_link_libraries(cppdl_test PRIVATE gtest gtest_main)

add_custom_target(check
    COMMAND $<TARGET_FILE:cppdl_test>
    DEPENDS cppdl_test
    USES_TERMINAL)
