cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

enable_testing()

project(
  cppdl
  LANGUAGES CXX
  VERSION 0.1.0)

# Workaround for MSVC on Windows
# https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

option(CPPDL_IREE_PATH "Path to the IREE installation." "")
if(CPPDL_IREE_PATH)
  set(CPPDL_USE_IREE
      ON
      CACHE BOOL "Use IREE for compilation" FORCE)

  get_filename_component(CPPDL_IREE_PATH_ABSOLUTE "${CPPDL_IREE_PATH}" ABSOLUTE)

  set(CPPDL_IREE_COMPILER_LIB
      "${CPPDL_IREE_PATH_ABSOLUTE}/lib/libIREECompiler.so"
      CACHE FILEPATH "Path to the IREE Compiler library")

  list(APPEND CMAKE_PREFIX_PATH "${CPPDL_IREE_PATH_ABSOLUTE}/lib/cmake/IREE")
  find_package(Threads REQUIRED)
  find_package(IREECompiler REQUIRED)
endif()

macro(add_cppdl_target NAME)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${NAME} PRIVATE -Wall -Wextra -Wpedantic
                                           -march=native)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${NAME} PRIVATE /W4)
  endif()
  target_compile_features(${NAME} PUBLIC cxx_std_17)
  target_include_directories(${NAME}
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  if(CPPDL_USE_IREE)
    target_compile_definitions(
      ${NAME} PRIVATE CPPDL_IREE_COMPILER_LIB="${CPPDL_IREE_COMPILER_LIB}")
  endif()
  target_link_libraries(${NAME} PRIVATE fmt::fmt)
endmacro()

macro(add_cppdl_library NAME)
  add_library(${NAME} ${ARGN})
  add_cppdl_target(${NAME})
endmacro()

macro(add_cppdl_executable NAME)
  add_executable(${NAME} ${ARGN})
  add_cppdl_target(${NAME})
  target_link_libraries(${NAME} PRIVATE cppdl)
endmacro()

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(unittests)
add_subdirectory(vendor/googletest)
add_subdirectory(vendor/fmt)

get_filename_component(SRC_DIR_ABSOLUTE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                       ABSOLUTE)
get_filename_component(UNITTESTS_DIR_ABSOLUTE
                       "${CMAKE_CURRENT_SOURCE_DIR}/unittests" ABSOLUTE)

add_custom_target(
  tidy
  COMMAND run-clang-tidy ${SRC_DIR_ABSOLUTE} ${UNITTESTS_DIR_ABSOLUTE}
          -use-color 1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  USES_TERMINAL)
