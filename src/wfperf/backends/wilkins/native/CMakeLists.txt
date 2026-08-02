# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

cmake_minimum_required(VERSION 3.15)
project(wfperf_wilkins_modules LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS C)

if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "WfPerf Wilkins modules require parallel HDF5")
endif()

find_path(HENSON_INCLUDE_DIR
    NAMES henson/context.h
    HINTS ${HENSON_ROOT} ENV HENSON_ROOT
    PATH_SUFFIXES include
    REQUIRED)
find_library(HENSON_LIBRARY
    NAMES henson
    HINTS ${HENSON_ROOT} ENV HENSON_ROOT
    PATH_SUFFIXES lib lib64
    REQUIRED)
find_library(HENSON_PMPI_LIBRARY
    NAMES henson-pmpi-static henson-pmpi
    HINTS ${HENSON_ROOT} ENV HENSON_ROOT
    PATH_SUFFIXES lib lib64
    REQUIRED)

function(add_wfperf_module target source)
    add_library(${target} SHARED ${source})
    target_include_directories(${target} PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${HENSON_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE
        MPI::MPI_CXX
        ${HDF5_LIBRARIES}
        ${HENSON_LIBRARY}
        ${HENSON_PMPI_LIBRARY})
    set_target_properties(${target} PROPERTIES PREFIX "" SUFFIX ".hx")

    if(APPLE)
        target_link_options(${target} PRIVATE
            "-Wl,-u,_henson_set_contexts,-u,_henson_set_namemap")
    else()
        target_link_options(${target} PRIVATE
            "-pie"
            "-Wl,--export-dynamic"
            "-Wl,-u,henson_set_contexts,-u,henson_set_namemap")
    endif()
endfunction()

add_wfperf_module(prod-henson prod-henson.cpp)
add_wfperf_module(middle middle.cpp)
add_wfperf_module(con-henson con-henson.cpp)

install(TARGETS prod-henson middle con-henson
        LIBRARY DESTINATION libexec/wfperf/wilkins
        RUNTIME DESTINATION libexec/wfperf/wilkins)
