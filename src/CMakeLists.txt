add_executable(inference
	${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp
)

add_executable(perceptron
	${CMAKE_CURRENT_SOURCE_DIR}/perceptron.cpp
)

add_executable(perceptron_simple
	${CMAKE_CURRENT_SOURCE_DIR}/perceptron_simple.cpp
)

foreach(TARGET inference perceptron perceptron_simple)
	target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -march=native -g)
	target_compile_features(${TARGET} PUBLIC cxx_std_17)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(${TARGET} PRIVATE fmt::fmt)
endforeach()
