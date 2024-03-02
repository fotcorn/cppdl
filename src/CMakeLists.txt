add_executable(mnist
	${CMAKE_CURRENT_SOURCE_DIR}/mnist.cpp
)

add_executable(graph
	${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
)

add_executable(codegen
	${CMAKE_CURRENT_SOURCE_DIR}/codegen.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/cppdl/tinymlir.cpp
)

foreach(TARGET mnist graph codegen)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -march=native)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		target_compile_options(${TARGET} PRIVATE /W4)
	endif()
	target_compile_features(${TARGET} PUBLIC cxx_std_17)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
	target_link_libraries(${TARGET} PRIVATE fmt::fmt)
endforeach()
