cmake_minimum_required(VERSION 3.14)
project(ba VERSION 0.1.0 LANGUAGES CXX)

# Global C++ Setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-march=native -O3 -Wall -pedantic)

# Dependencies
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
add_subdirectory(deps/unordered_dense)
add_subdirectory(deps/lgmath)

# Library
add_subdirectory(src)
target_link_libraries(ba unordered_dense::unordered_dense lgmath yaml-cpp Eigen3::Eigen ${OpenCV_LIBS} Threads::Threads OpenMP::OpenMP_CXX)

# Enable testing
enable_testing()

# Link executable directories
add_subdirectory(tests)