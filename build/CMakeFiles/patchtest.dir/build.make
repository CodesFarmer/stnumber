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
include CMakeFiles/patchtest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/patchtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/patchtest.dir/flags.make

CMakeFiles/patchtest.dir/test/test_patch.cc.o: CMakeFiles/patchtest.dir/flags.make
CMakeFiles/patchtest.dir/test/test_patch.cc.o: ../test/test_patch.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lowell/MachineLearning/CppProgram/CaffeTools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/patchtest.dir/test/test_patch.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/patchtest.dir/test/test_patch.cc.o -c /home/lowell/MachineLearning/CppProgram/CaffeTools/test/test_patch.cc

CMakeFiles/patchtest.dir/test/test_patch.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/patchtest.dir/test/test_patch.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lowell/MachineLearning/CppProgram/CaffeTools/test/test_patch.cc > CMakeFiles/patchtest.dir/test/test_patch.cc.i

CMakeFiles/patchtest.dir/test/test_patch.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/patchtest.dir/test/test_patch.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lowell/MachineLearning/CppProgram/CaffeTools/test/test_patch.cc -o CMakeFiles/patchtest.dir/test/test_patch.cc.s

CMakeFiles/patchtest.dir/test/test_patch.cc.o.requires:

.PHONY : CMakeFiles/patchtest.dir/test/test_patch.cc.o.requires

CMakeFiles/patchtest.dir/test/test_patch.cc.o.provides: CMakeFiles/patchtest.dir/test/test_patch.cc.o.requires
	$(MAKE) -f CMakeFiles/patchtest.dir/build.make CMakeFiles/patchtest.dir/test/test_patch.cc.o.provides.build
.PHONY : CMakeFiles/patchtest.dir/test/test_patch.cc.o.provides

CMakeFiles/patchtest.dir/test/test_patch.cc.o.provides.build: CMakeFiles/patchtest.dir/test/test_patch.cc.o


# Object files for target patchtest
patchtest_OBJECTS = \
"CMakeFiles/patchtest.dir/test/test_patch.cc.o"

# External object files for target patchtest
patchtest_EXTERNAL_OBJECTS =

patchtest: CMakeFiles/patchtest.dir/test/test_patch.cc.o
patchtest: CMakeFiles/patchtest.dir/build.make
patchtest: /usr/local/lib/libtinyxml2.so
patchtest: libctools.so
patchtest: /usr/local/lib/libopencv_videostab.so.3.1.0
patchtest: /usr/local/lib/libopencv_superres.so.3.1.0
patchtest: /usr/local/lib/libopencv_stitching.so.3.1.0
patchtest: /usr/local/lib/libopencv_shape.so.3.1.0
patchtest: /usr/local/lib/libopencv_video.so.3.1.0
patchtest: /usr/local/lib/libopencv_photo.so.3.1.0
patchtest: /usr/local/lib/libopencv_objdetect.so.3.1.0
patchtest: /usr/local/lib/libopencv_calib3d.so.3.1.0
patchtest: /usr/local/lib/libopencv_features2d.so.3.1.0
patchtest: /usr/local/lib/libopencv_ml.so.3.1.0
patchtest: /usr/local/lib/libopencv_flann.so.3.1.0
patchtest: /home/lowell/softwares/caffe/build/install/lib/libcaffe.so.1.0.0
patchtest: /usr/local/lib/libopencv_highgui.so.3.1.0
patchtest: /usr/local/lib/libopencv_videoio.so.3.1.0
patchtest: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
patchtest: /usr/local/lib/libopencv_imgproc.so.3.1.0
patchtest: /usr/local/lib/libopencv_core.so.3.1.0
patchtest: /home/lowell/softwares/caffe/build/install/lib/libcaffeproto.a
patchtest: /usr/lib/x86_64-linux-gnu/libboost_system.so
patchtest: /usr/lib/x86_64-linux-gnu/libboost_thread.so
patchtest: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
patchtest: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
patchtest: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
patchtest: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
patchtest: /usr/lib/x86_64-linux-gnu/libglog.so
patchtest: /usr/lib/x86_64-linux-gnu/libgflags.so
patchtest: /usr/lib/x86_64-linux-gnu/libprotobuf.so
patchtest: /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so
patchtest: /usr/lib/x86_64-linux-gnu/libhdf5.so
patchtest: /usr/lib/x86_64-linux-gnu/libpthread.so
patchtest: /usr/lib/x86_64-linux-gnu/libz.so
patchtest: /usr/lib/x86_64-linux-gnu/libdl.so
patchtest: /usr/lib/x86_64-linux-gnu/libm.so
patchtest: /usr/lib/x86_64-linux-gnu/libhdf5_hl_cpp.so
patchtest: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
patchtest: /usr/lib/x86_64-linux-gnu/liblmdb.so
patchtest: /usr/lib/x86_64-linux-gnu/libleveldb.so
patchtest: /usr/lib/liblapack.so
patchtest: /usr/lib/libcblas.so
patchtest: /usr/lib/libatlas.so
patchtest: /usr/lib/x86_64-linux-gnu/libboost_python.so
patchtest: /usr/local/lib/libtinyxml2.so
patchtest: CMakeFiles/patchtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lowell/MachineLearning/CppProgram/CaffeTools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable patchtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/patchtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/patchtest.dir/build: patchtest

.PHONY : CMakeFiles/patchtest.dir/build

CMakeFiles/patchtest.dir/requires: CMakeFiles/patchtest.dir/test/test_patch.cc.o.requires

.PHONY : CMakeFiles/patchtest.dir/requires

CMakeFiles/patchtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/patchtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/patchtest.dir/clean

CMakeFiles/patchtest.dir/depend:
	cd /home/lowell/MachineLearning/CppProgram/CaffeTools/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lowell/MachineLearning/CppProgram/CaffeTools /home/lowell/MachineLearning/CppProgram/CaffeTools /home/lowell/MachineLearning/CppProgram/CaffeTools/build /home/lowell/MachineLearning/CppProgram/CaffeTools/build /home/lowell/MachineLearning/CppProgram/CaffeTools/build/CMakeFiles/patchtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/patchtest.dir/depend

