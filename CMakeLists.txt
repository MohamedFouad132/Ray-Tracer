cmake_minimum_required(VERSION 3.18)
project(cuTrace LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(raytracer main.cu render.cu)

target_link_libraries(raytracer PRIVATE curand)