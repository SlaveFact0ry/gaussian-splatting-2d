cmake_minimum_required(VERSION 3.18)
project(render_2dgs_cuda LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 86)          # RTX 3090

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(
    render_2dgs_cuda_bindings
    STABLE_ABI
    src/render_2dgs_cuda_bindings.cpp
    src/render_forward.cu
    # v2에서 추가:
    # src/device_buffers.cu
    # src/tile_binning.cu
)

set_target_properties(render_2dgs_cuda_bindings PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS render_2dgs_cuda_bindings DESTINATION ${CMAKE_SOURCE_DIR})