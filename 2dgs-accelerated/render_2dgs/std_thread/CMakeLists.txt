cmake_minimum_required(VERSION 3.15)
project(render_2dgs_std_thread LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

message(STATUS "Python: ${Python_EXECUTABLE} (version ${Python_VERSION})")

nanobind_add_module(
    std_thread_bindings
    STABLE_ABI
    src/render_2dgs_bindings.cpp
    src/render_forward.cpp
    src/render_backward.cpp
    src/thread_pool.cpp
)

install(TARGETS std_thread_bindings DESTINATION ${CMAKE_SOURCE_DIR})

message(STATUS "")
message(STATUS "=== nanobind Render 2DGS std_thread Backend ===")
message(STATUS "After building, the module will be in build/")
message(STATUS "Then: python3 -c 'import std_thread_bindings; print(std_thread_bindings.__doc__)'")
message(STATUS "")
