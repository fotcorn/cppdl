cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

enable_testing()

project(cppdl
	LANGUAGES CXX
	VERSION 0.1.0
)

# Workaround for MSVC on Windows
# https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(src)
add_subdirectory(unittests)
add_subdirectory(vendor/googletest)
add_subdirectory(vendor/fmt)

get_filename_component(SRC_DIR_ABSOLUTE "${CMAKE_CURRENT_SOURCE_DIR}/src" ABSOLUTE)
get_filename_component(UNITTESTS_DIR_ABSOLUTE "${CMAKE_CURRENT_SOURCE_DIR}/unittests" ABSOLUTE)


add_custom_target(
    tidy
    COMMAND run-clang-tidy ${SRC_DIR_ABSOLUTE} ${UNITTESTS_DIR_ABSOLUTE} -use-color 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	USES_TERMINAL
)
