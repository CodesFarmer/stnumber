cmake_minimum_required(VERSION 3.1.0)

add_definitions(-DUSE_OPENCV)
#add_definitions(-DUSE_DEPTH)
#add_definitions(-DINPUT_L2NORM)
include(cmake/ConfigureOpenCV.cmake)
include(cmake/ConfigureCaffe.cmake)
include(cmake/ConfigureTinyxml2.cmake)
include(cmake/ConfigureHDF5.cmake)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

file(GLOB header_dir
        "include/*")
include_directories(${header_dir})
include_directories(${DEST_INCLUDE_DIRS})

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

file(GLOB models
	"data/model/*.caffemodel"
	"data/model/*.prototxt"
)
set(CMAKE_MODEL_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/data/model)
install(FILES ${models} DESTINATION ${CMAKE_MODEL_INSTALL_PATH})
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
add_executable(hdf5test test/test_hdf5.cc)
add_executable(gstest test/gesture_data.cc)
target_link_libraries(printtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(xmltest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(pathtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(patchtest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(mtcnntest ${DEST_LINK_TARGETS} ctools)
target_link_libraries(bbxtest ${DEST_LINK_TARGETS} handdetect)
target_link_libraries(hdf5test ${DEST_LINK_TARGETS} ctools)
target_link_libraries(gstest ${DEST_LINK_TARGETS} ctools)
