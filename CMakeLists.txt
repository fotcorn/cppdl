cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

enable_testing()

project(cppdl
	LANGUAGES CXX
	VERSION 0.1.0
)

add_subdirectory(src)
add_subdirectory(unittests)
add_subdirectory(vendor/googletest)
add_subdirectory(vendor/fmt)
