cmake_minimum_required(VERSION 3.1.0)

include(cmake/ConfigureOpenCV.cmake)
include(cmake/ConfigureCaffe.cmake)
include(cmake/ConfigureTinyxml2.cmake)

file(GLOB header_dir
        "include/*")
include_directories(${header_dir})

file(GLOB ctools_header
        "include/common/*.h"
        "include/detection/*.h"
        "include/cnnetwork/*.h")
file(GLOB ctools_source
        "src/common/*.cc"
        "src/detection/*.cc")

add_library(ctools
        ${ctools_header}
        ${ctools_source}
        )
target_link_libraries(ctools ${DEST_LINK_TARGETS})

add_executable(printtest test/test_fp.cc)
add_executable(xmltest test/test_xml.cc)
add_executable(pathtest test/test_patch.cc)
add_executable(patchtest test/test_patch.cc)
target_link_libraries(printtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(xmltest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(pathtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(patchtest ${DEST_LINK_TARGETS} ctools)
