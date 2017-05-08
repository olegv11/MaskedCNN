cmake_minimum_required (VERSION 2.8)

project (MaskedCNN)

find_package( OpenCV REQUIRED )
find_library( OpenBLAS openblas )

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})


set(CMAKE_CXX_STANDARD 14)
set(GCC_FLAGS "-Wall -Wextra -pedantic -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fno-omit-frame-pointer")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")
set(CMAKE_VERBOSE_MAKEFILE TRUE)

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/maskedcnn/*.h*)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include/maskedcnn")

add_executable(maskedcnnexe
    ${sources} ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp ${headers})
target_link_libraries(maskedcnnexe ${OpenCV_LIBS} ${OpenBLAS} ${PROTOBUF_LIBRARY})


add_library(maskedcnn
    ${sources} ${headers})
SET_TARGET_PROPERTIES(maskedcnn PROPERTIES COMPILE_FLAGS "-fPIC")
target_link_libraries(maskedcnn ${OpenCV_LIBS} ${OpenBLAS} ${PROTOBUF_LIBRARY})

add_subdirectory(test)

install(TARGETS maskedcnn maskedcnnexe
        ARCHIVE DESTINATION lib/maskedcnn
        LIBRARY DESTINATION lib/maskedcnn
        RUNTIME DESTINATION bin)
