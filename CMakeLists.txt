cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 120)

project(mini_cuda-ops LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add src/vector_add.cu)
