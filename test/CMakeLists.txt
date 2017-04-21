# We need thread support
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(testmaskedcnn ${sources})
target_link_libraries(testmaskedcnn ${GTEST_LIBRARIES} libmaskedcnn pthread)
