cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(cppdl
	LANGUAGES CXX
	VERSION 0.1.0
)

add_executable(cppdl
	${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_compile_options(cppdl PRIVATE -Wall -Wextra -pedantic -march=native -g)
target_compile_features(cppdl PUBLIC cxx_std_17)
set_target_properties(cppdl PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(cppdl PROPERTIES CXX_EXTENSIONS OFF)
