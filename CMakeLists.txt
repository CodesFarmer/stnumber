cmake_minimum_required(VERSION 3.1.0)

include(cmake/ConfigureOpenCV.cmake)
include(cmake/ConfigureCaffe.cmake)
include(cmake/ConfigureTinyxml2.cmake)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

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

add_library(ctools SHARED
        ${ctools_header}
        ${ctools_source}
        )
target_link_libraries(ctools ${DEST_LINK_TARGETS})

set(SOURCE_HH_FILE
        "include/detection/hand_boundingbox.h")
set(SOURCE_CC_FILE
        "src/detection/hand_boundingbox.cc"
        )
add_library(handdetect SHARED
        ${SOURCE_HH_FILE}
        ${SOURCE_CC_FILE}
        )
set_target_properties(handdetect PROPERTIES OUTPUT_NAME "handdetect")
target_link_libraries(handdetect ${DEST_LINK_TARGETS} ctools)

set(CT_INCLUDE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
set(CT_LIB_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_RPATH ${CT_LIB_INSTALL_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
install(TARGETS ctools DESTINATION ${CMAKE_INSTALL_RPATH})
install(TARGETS handdetect DESTINATION ${CMAKE_INSTALL_RPATH})
install(DIRECTORY include DESTINATION ${CT_INCLUDE_INSTALL_PATH})

message(STATUS "The install path: ${CMAKE_INSTALL_PREFIX}")

add_executable(printtest test/test_fp.cc)
add_executable(xmltest test/test_xml.cc)
add_executable(pathtest test/test_patch.cc)
add_executable(patchtest test/test_patch.cc)
add_executable(mtcnntest test/test_mtcnn.cc)
add_executable(bbxtest test/test_bbx.cc)
target_link_libraries(printtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(xmltest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(pathtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(patchtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(mtcnntest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(bbxtest handdetect)