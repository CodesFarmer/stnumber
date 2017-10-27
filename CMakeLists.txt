cmake_minimum_required(VERSION 3.1.0)

include(cmake/ConfigureOpenCV.cmake)
include(cmake/ConfigureCaffe.cmake)

file(GLOB header_dir
        "include/*")
include_directories(${header_dir})

file(GLOB dependencies_header
        "include/detection/*.h"
        "include/common/*.h"
        "include/cnnetwork/*.h")
file(GLOB dependencies_source
        "src/detection/*.cc"
        "src/common/*.cc")

add_library(plibrary
        ${dependencies_header}
        ${dependencies_source}
)

add_executable(printtest test/test.cc)
add_executable(cnntest test/test_cnn.cc)
target_link_libraries(printtest plibrary ${DEST_LINK_TARGETS})
target_link_libraries(cnntest plibrary ${DEST_LINK_TARGETS})
