# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lowell/MachineLearning/CppProgram/CaffeTools

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lowell/MachineLearning/CppProgram/CaffeTools/build

# Include any dependencies generated for this target.
include CMakeFiles/hdf5test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hdf5test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hdf5test.dir/flags.make

CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o: CMakeFiles/hdf5test.dir/flags.make
CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o: ../test/test_hdf5.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lowell/MachineLearning/CppProgram/CaffeTools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o -c /home/lowell/MachineLearning/CppProgram/CaffeTools/test/test_hdf5.cc

CMakeFiles/hdf5test.dir/test/test_hdf5.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hdf5test.dir/test/test_hdf5.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lowell/MachineLearning/CppProgram/CaffeTools/test/test_hdf5.cc > CMakeFiles/hdf5test.dir/test/test_hdf5.cc.i

CMakeFiles/hdf5test.dir/test/test_hdf5.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hdf5test.dir/test/test_hdf5.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lowell/MachineLearning/CppProgram/CaffeTools/test/test_hdf5.cc -o CMakeFiles/hdf5test.dir/test/test_hdf5.cc.s

CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.requires:

.PHONY : CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.requires

CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.provides: CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.requires
	$(MAKE) -f CMakeFiles/hdf5test.dir/build.make CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.provides.build
.PHONY : CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.provides

CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.provides.build: CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o


# Object files for target hdf5test
hdf5test_OBJECTS = \
"CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o"

# External object files for target hdf5test
hdf5test_EXTERNAL_OBJECTS =

hdf5test: CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o
hdf5test: CMakeFiles/hdf5test.dir/build.make
hdf5test: /usr/local/lib/libtinyxml2.so
hdf5test: libctools.so
hdf5test: /usr/local/lib/libopencv_videostab.so.3.1.0
hdf5test: /usr/local/lib/libopencv_superres.so.3.1.0
hdf5test: /usr/local/lib/libopencv_stitching.so.3.1.0
hdf5test: /usr/local/lib/libopencv_shape.so.3.1.0
hdf5test: /usr/local/lib/libopencv_video.so.3.1.0
hdf5test: /usr/local/lib/libopencv_photo.so.3.1.0
hdf5test: /usr/local/lib/libopencv_objdetect.so.3.1.0
hdf5test: /usr/local/lib/libopencv_calib3d.so.3.1.0
hdf5test: /usr/local/lib/libopencv_features2d.so.3.1.0
hdf5test: /usr/local/lib/libopencv_ml.so.3.1.0
hdf5test: /usr/local/lib/libopencv_flann.so.3.1.0
hdf5test: /home/lowell/softwares/caffe/build/install/lib/libcaffe.so.1.0.0
hdf5test: /usr/local/lib/libopencv_highgui.so.3.1.0
hdf5test: /usr/local/lib/libopencv_videoio.so.3.1.0
hdf5test: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
hdf5test: /usr/local/lib/libopencv_imgproc.so.3.1.0
hdf5test: /usr/local/lib/libopencv_core.so.3.1.0
hdf5test: /home/lowell/softwares/caffe/build/install/lib/libcaffeproto.a
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_system.so
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_thread.so
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
hdf5test: /usr/lib/x86_64-linux-gnu/libglog.so
hdf5test: /usr/lib/x86_64-linux-gnu/libgflags.so
hdf5test: /usr/lib/x86_64-linux-gnu/libprotobuf.so
hdf5test: /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so
hdf5test: /usr/lib/x86_64-linux-gnu/libhdf5.so
hdf5test: /usr/lib/x86_64-linux-gnu/libpthread.so
hdf5test: /usr/lib/x86_64-linux-gnu/libz.so
hdf5test: /usr/lib/x86_64-linux-gnu/libdl.so
hdf5test: /usr/lib/x86_64-linux-gnu/libm.so
hdf5test: /usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so
hdf5test: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
hdf5test: /usr/lib/x86_64-linux-gnu/liblmdb.so
hdf5test: /usr/lib/x86_64-linux-gnu/libleveldb.so
hdf5test: /usr/lib/liblapack.so
hdf5test: /usr/lib/libcblas.so
hdf5test: /usr/lib/libatlas.so
hdf5test: /usr/lib/x86_64-linux-gnu/libboost_python.so
hdf5test: /usr/local/lib/libtinyxml2.so
hdf5test: CMakeFiles/hdf5test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lowell/MachineLearning/CppProgram/CaffeTools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hdf5test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hdf5test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hdf5test.dir/build: hdf5test

.PHONY : CMakeFiles/hdf5test.dir/build

CMakeFiles/hdf5test.dir/requires: CMakeFiles/hdf5test.dir/test/test_hdf5.cc.o.requires

.PHONY : CMakeFiles/hdf5test.dir/requires

CMakeFiles/hdf5test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hdf5test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hdf5test.dir/clean

CMakeFiles/hdf5test.dir/depend:
	cd /home/lowell/MachineLearning/CppProgram/CaffeTools/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lowell/MachineLearning/CppProgram/CaffeTools /home/lowell/MachineLearning/CppProgram/CaffeTools /home/lowell/MachineLearning/CppProgram/CaffeTools/build /home/lowell/MachineLearning/CppProgram/CaffeTools/build /home/lowell/MachineLearning/CppProgram/CaffeTools/build/CMakeFiles/hdf5test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hdf5test.dir/depend

