add_executable(cppdl_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_elementwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_slice.cpp
)

target_include_directories(cppdl_test PRIVATE ../src)
target_compile_options(cppdl_test PRIVATE -Wall -Wextra -pedantic -march=native -g)
target_compile_features(cppdl_test PUBLIC cxx_std_17)
target_link_libraries(cppdl_test gtest gtest_main)

add_test(NAME cppdl_test COMMAND cppdl_test)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS cppdl_test)
