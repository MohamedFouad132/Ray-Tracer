cmake_minimum_required(VERSION 3.18)
project(cuTrace LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(raytracer src/main.cu src/render.cu)

target_include_directories(raytracer PRIVATE include)
target_link_libraries(raytracer PRIVATE curand)