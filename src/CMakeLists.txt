add_executable(inference
	${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp
)

foreach(TARGET inference)
	target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -march=native -g)
	target_compile_features(${TARGET} PUBLIC cxx_std_17)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
