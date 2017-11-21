# Install script for directory: /home/slam/DepthGesture/stnumber

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/slam/DepthGesture/stnumber/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/slam/DepthGesture/stnumber/build/install/data/model/onet.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/onet_dpir.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/pnet.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/pnet_dpir.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/rnet.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/rnet_dpir.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/tnet.caffemodel;/home/slam/DepthGesture/stnumber/build/install/data/model/onet_deploy.prototxt;/home/slam/DepthGesture/stnumber/build/install/data/model/pnet_deploy.prototxt;/home/slam/DepthGesture/stnumber/build/install/data/model/rnet_deploy.prototxt;/home/slam/DepthGesture/stnumber/build/install/data/model/tnet_deploy.prototxt")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/slam/DepthGesture/stnumber/build/install/data/model" TYPE FILE FILES
    "/home/slam/DepthGesture/stnumber/data/model/onet.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/onet_dpir.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/pnet.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/pnet_dpir.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/rnet.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/rnet_dpir.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/tnet.caffemodel"
    "/home/slam/DepthGesture/stnumber/data/model/onet_deploy.prototxt"
    "/home/slam/DepthGesture/stnumber/data/model/pnet_deploy.prototxt"
    "/home/slam/DepthGesture/stnumber/data/model/rnet_deploy.prototxt"
    "/home/slam/DepthGesture/stnumber/data/model/tnet_deploy.prototxt"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/slam/DepthGesture/stnumber/build/install/lib" TYPE SHARED_LIBRARY FILES "/home/slam/DepthGesture/stnumber/build/libctools.so")
  if(EXISTS "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so"
         OLD_RPATH "/home/slam/UntouchWorkspace/Thirdparty/free/x86_64-linux/opencv-3.1.0/build/lib:/home/slam/UntouchWorkspace/Thirdparty/free/x86_64-linux/caffe/build/install/lib:/usr/local/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libctools.so")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/slam/DepthGesture/stnumber/build/install/lib" TYPE SHARED_LIBRARY FILES "/home/slam/DepthGesture/stnumber/build/libhanddetect.so")
  if(EXISTS "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so"
         OLD_RPATH "/home/slam/UntouchWorkspace/Thirdparty/free/x86_64-linux/opencv-3.1.0/build/lib:/usr/local/lib:/home/slam/DepthGesture/stnumber/build:/home/slam/UntouchWorkspace/Thirdparty/free/x86_64-linux/caffe/build/install/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/slam/DepthGesture/stnumber/build/install/lib/libhanddetect.so")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/slam/DepthGesture/stnumber/build/install/include")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/slam/DepthGesture/stnumber/build/install" TYPE DIRECTORY FILES "/home/slam/DepthGesture/stnumber/include")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/slam/DepthGesture/stnumber/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
