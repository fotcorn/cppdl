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
