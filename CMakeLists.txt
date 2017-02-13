cmake_minimum_required (VERSION 2.6)
project (MaskedCNN)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 14)
set(GCC_FLAGS "-Wall -Wextra -pedantic -O2 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")

file(GLOB sources ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_SOURCE_DIR}/include/maskedcnn/*.hpp)

include_directories("${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/maskedcnn")

add_executable(maskedcnn
    ${sources} ${headers})


