add_executable(inference
	${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp
)

add_executable(perceptron
	${CMAKE_CURRENT_SOURCE_DIR}/perceptron.cpp
)

add_executable(nn
	${CMAKE_CURRENT_SOURCE_DIR}/nn.cpp
)

foreach(TARGET inference perceptron nn)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -march=native)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		target_compile_options(${TARGET} PRIVATE /W4)
	endif()
	target_compile_features(${TARGET} PUBLIC cxx_std_17)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(${TARGET} PRIVATE fmt::fmt)
endforeach()
