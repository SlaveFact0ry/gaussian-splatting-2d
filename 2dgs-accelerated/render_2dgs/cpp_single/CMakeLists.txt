cmake_minimum_required(VERSION 3.15)
project(render_nanobind LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and nanobind
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Download nanobind if not found (using FetchContent)
include(FetchContent)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

message(STATUS "Python: ${Python_EXECUTABLE}")

# Build the nanobind module
nanobind_add_module(
    render_2dgs_bindings
    STABLE_ABI
    src/render_2dgs_bindings.cpp
    src/render_forward.cpp
    src/render_backward.cpp
)

# Install to current directory for easy testing
install(TARGETS render_2dgs_bindings DESTINATION ${CMAKE_SOURCE_DIR})

message(STATUS "")
message(STATUS "=== nanobind Render 2DGS Bindings ===")
message(STATUS "After building, the module will be in build/")
message(STATUS "Copy render_2dgs_bindings*.so to your Python path or use:")
message(STATUS "  export PYTHONPATH=\${PYTHONPATH}:$(pwd)/build")
message(STATUS "Then: python3 -c 'import render_2dgs_bindings; print(render_2dgs_bindings.__doc__)'")
message(STATUS "")
