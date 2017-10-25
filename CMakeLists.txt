cmake_minimum_required(VERSION 3.1.0)

include(ConfigureOpenCV.cmake)
include(ConfigureCaffe.cmake)

file(GLOB dependencies_header
        "include/detection/*.h")
file(GLOB dependencies_source
        "include/detection/*.cc")
