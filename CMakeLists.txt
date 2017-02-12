cmake_minimum_required (VERSION 2.6)
project (MaskedCNN)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 14)

file(GLOB sources ${CMAKE_SOURCE_DIR}/src/*.cpp)

include_directories(include/maskedcnn)

add_executable(maskedcnn
    ${sources})


