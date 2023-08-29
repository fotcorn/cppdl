add_executable(cppdl
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_options(cppdl PRIVATE -Wall -Wextra -pedantic -march=native -g)
target_compile_features(cppdl PUBLIC cxx_std_17)
