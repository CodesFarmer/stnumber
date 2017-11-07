set(CaffePath "/home/lowell/softwares/caffe")
#set(CaffePath "/home/slam/FaceUntouch/caffe-master")
LIST(APPEND DEST_INCLUDE_DIRS "${CaffePath}/include")
set(WITH_CAFFE "Build with Caffe support" ON)

if(WITH_CAFFE)
    set(CMAKE_PREFIX_PATH "${CaffePath}/build/install/share/Caffe;${CMAKE_PREFIX_PATH}")
    find_package(Caffe REQUIRED)
    list(APPEND DEST_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
    list(APPEND DEST_INCLUDE_DIRS "${CaffePath}/build/include")
    list(APPEND DEST_LINK_TARGETS ${Caffe_LIBRARIES})
    message(STATUS ${Caffe_INCLUDE_DIRS})
    message(STATUS "Compile with Caffe")
    message(STATUS "The Search Path is ${Caffe_LIBRARIES}")
else()
    message(STATUS "Compile without Caffe")
endif()

list(APPEND DEST_INCLUDE_DIRS "include")
#include_directories(${DEST_INCLUDE_DIRS})
