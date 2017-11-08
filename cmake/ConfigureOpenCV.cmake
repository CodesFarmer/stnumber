set(CMAKE_PREFIX_PATH "$ENV{UNTOUCH_LOCAL}/Thirdparty/free/x86_64-linux/opencv-3.1.0/build;${CMAKE_PREFIX_PATH}")
set(WITH_OPENCV ON "Compile with OpenCV")
set(USE_OPENCV "Build with OpenCV support" ON)
find_package(OpenCV)

if(WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    list(APPEND ALL_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    list(APPEND DEST_LINK_TARGETS ${OpenCV_LIBRARIES})
    message(STATUS "Compile with OpenCV")
else()
    message(STATUS "Compile without OpenCV")
endif()

