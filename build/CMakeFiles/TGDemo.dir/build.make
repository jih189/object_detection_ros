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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiaming/object_tracking_2D

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiaming/object_tracking_2D/build

# Include any dependencies generated for this target.
include CMakeFiles/TGDemo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TGDemo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TGDemo.dir/flags.make

CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o: ../include/object_tracking_2D/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o -c /home/jiaming/object_tracking_2D/include/object_tracking_2D/utils.cpp

CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/include/object_tracking_2D/utils.cpp > CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.i

CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/include/object_tracking_2D/utils.cpp -o CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.s

CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.requires

CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.provides: CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.provides

CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.provides.build: CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o


CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o: ../src/templates_generator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o -c /home/jiaming/object_tracking_2D/src/templates_generator.cpp

CMakeFiles/TGDemo.dir/src/templates_generator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/templates_generator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/templates_generator.cpp > CMakeFiles/TGDemo.dir/src/templates_generator.cpp.i

CMakeFiles/TGDemo.dir/src/templates_generator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/templates_generator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/templates_generator.cpp -o CMakeFiles/TGDemo.dir/src/templates_generator.cpp.s

CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.requires

CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.provides: CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.provides

CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o


CMakeFiles/TGDemo.dir/src/Camera.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/Camera.cpp.o: ../src/Camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/TGDemo.dir/src/Camera.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/Camera.cpp.o -c /home/jiaming/object_tracking_2D/src/Camera.cpp

CMakeFiles/TGDemo.dir/src/Camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/Camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/Camera.cpp > CMakeFiles/TGDemo.dir/src/Camera.cpp.i

CMakeFiles/TGDemo.dir/src/Camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/Camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/Camera.cpp -o CMakeFiles/TGDemo.dir/src/Camera.cpp.s

CMakeFiles/TGDemo.dir/src/Camera.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/Camera.cpp.o.requires

CMakeFiles/TGDemo.dir/src/Camera.cpp.o.provides: CMakeFiles/TGDemo.dir/src/Camera.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/Camera.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/Camera.cpp.o.provides

CMakeFiles/TGDemo.dir/src/Camera.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/Camera.cpp.o


CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o: ../src/EdgeTracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o -c /home/jiaming/object_tracking_2D/src/EdgeTracker.cpp

CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/EdgeTracker.cpp > CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.i

CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/EdgeTracker.cpp -o CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.s

CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.requires

CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.provides: CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.provides

CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o


CMakeFiles/TGDemo.dir/src/epnp.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/epnp.cpp.o: ../src/epnp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/TGDemo.dir/src/epnp.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/epnp.cpp.o -c /home/jiaming/object_tracking_2D/src/epnp.cpp

CMakeFiles/TGDemo.dir/src/epnp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/epnp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/epnp.cpp > CMakeFiles/TGDemo.dir/src/epnp.cpp.i

CMakeFiles/TGDemo.dir/src/epnp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/epnp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/epnp.cpp -o CMakeFiles/TGDemo.dir/src/epnp.cpp.s

CMakeFiles/TGDemo.dir/src/epnp.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/epnp.cpp.o.requires

CMakeFiles/TGDemo.dir/src/epnp.cpp.o.provides: CMakeFiles/TGDemo.dir/src/epnp.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/epnp.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/epnp.cpp.o.provides

CMakeFiles/TGDemo.dir/src/epnp.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/epnp.cpp.o


CMakeFiles/TGDemo.dir/src/glm.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/glm.cpp.o: ../src/glm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/TGDemo.dir/src/glm.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/glm.cpp.o -c /home/jiaming/object_tracking_2D/src/glm.cpp

CMakeFiles/TGDemo.dir/src/glm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/glm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/glm.cpp > CMakeFiles/TGDemo.dir/src/glm.cpp.i

CMakeFiles/TGDemo.dir/src/glm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/glm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/glm.cpp -o CMakeFiles/TGDemo.dir/src/glm.cpp.s

CMakeFiles/TGDemo.dir/src/glm.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/glm.cpp.o.requires

CMakeFiles/TGDemo.dir/src/glm.cpp.o.provides: CMakeFiles/TGDemo.dir/src/glm.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/glm.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/glm.cpp.o.provides

CMakeFiles/TGDemo.dir/src/glm.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/glm.cpp.o


CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o: ../src/HomoTransform.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o -c /home/jiaming/object_tracking_2D/src/HomoTransform.cpp

CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/HomoTransform.cpp > CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.i

CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/HomoTransform.cpp -o CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.s

CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.requires

CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.provides: CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.provides

CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o


CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o: ../src/ModelImport.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o -c /home/jiaming/object_tracking_2D/src/ModelImport.cpp

CMakeFiles/TGDemo.dir/src/ModelImport.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/ModelImport.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/ModelImport.cpp > CMakeFiles/TGDemo.dir/src/ModelImport.cpp.i

CMakeFiles/TGDemo.dir/src/ModelImport.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/ModelImport.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/ModelImport.cpp -o CMakeFiles/TGDemo.dir/src/ModelImport.cpp.s

CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.requires

CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.provides: CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.provides

CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o


CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o: ../src/objectmodel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o -c /home/jiaming/object_tracking_2D/src/objectmodel.cpp

CMakeFiles/TGDemo.dir/src/objectmodel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/objectmodel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/objectmodel.cpp > CMakeFiles/TGDemo.dir/src/objectmodel.cpp.i

CMakeFiles/TGDemo.dir/src/objectmodel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/objectmodel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/objectmodel.cpp -o CMakeFiles/TGDemo.dir/src/objectmodel.cpp.s

CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.requires

CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.provides: CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.provides

CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o


CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o: ../src/ParticleFilter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o -c /home/jiaming/object_tracking_2D/src/ParticleFilter.cpp

CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/ParticleFilter.cpp > CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.i

CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/ParticleFilter.cpp -o CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.s

CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.requires

CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.provides: CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.provides

CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o


CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o: CMakeFiles/TGDemo.dir/flags.make
CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o: ../src/PoseEstimationSURF.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o -c /home/jiaming/object_tracking_2D/src/PoseEstimationSURF.cpp

CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaming/object_tracking_2D/src/PoseEstimationSURF.cpp > CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.i

CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaming/object_tracking_2D/src/PoseEstimationSURF.cpp -o CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.s

CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.requires:

.PHONY : CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.requires

CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.provides: CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.requires
	$(MAKE) -f CMakeFiles/TGDemo.dir/build.make CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.provides.build
.PHONY : CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.provides

CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.provides.build: CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o


# Object files for target TGDemo
TGDemo_OBJECTS = \
"CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o" \
"CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o" \
"CMakeFiles/TGDemo.dir/src/Camera.cpp.o" \
"CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o" \
"CMakeFiles/TGDemo.dir/src/epnp.cpp.o" \
"CMakeFiles/TGDemo.dir/src/glm.cpp.o" \
"CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o" \
"CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o" \
"CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o" \
"CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o" \
"CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o"

# External object files for target TGDemo
TGDemo_EXTERNAL_OBJECTS =

TGDemo: CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/Camera.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/epnp.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/glm.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o
TGDemo: CMakeFiles/TGDemo.dir/build.make
TGDemo: /usr/lib/x86_64-linux-gnu/libGL.so
TGDemo: /usr/lib/x86_64-linux-gnu/libGLU.so
TGDemo: /usr/lib/x86_64-linux-gnu/libGLEW.so
TGDemo: /usr/lib/x86_64-linux-gnu/libglut.so
TGDemo: /usr/lib/x86_64-linux-gnu/libXmu.so
TGDemo: /usr/lib/x86_64-linux-gnu/libXi.so
TGDemo: /usr/lib/liblapack.so
TGDemo: /usr/lib/libblas.so
TGDemo: /usr/local/lib/libopencv_videostab.so.2.4.9
TGDemo: /usr/local/lib/libopencv_ts.a
TGDemo: /usr/local/lib/libopencv_superres.so.2.4.9
TGDemo: /usr/local/lib/libopencv_stitching.so.2.4.9
TGDemo: /usr/local/lib/libopencv_contrib.so.2.4.9
TGDemo: 3rdparty/Fdcm/libFdcmLib.so
TGDemo: 3rdparty/Fitline/libLineFitLib.so
TGDemo: 3rdparty/Image/libImageLib.so
TGDemo: /home/jiaming/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so.1.14.0
TGDemo: /home/jiaming/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
TGDemo: /home/jiaming/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so.1
TGDemo: /usr/local/lib/libopencv_nonfree.so.2.4.9
TGDemo: /usr/local/lib/libopencv_ocl.so.2.4.9
TGDemo: /usr/local/lib/libopencv_gpu.so.2.4.9
TGDemo: /usr/local/lib/libopencv_photo.so.2.4.9
TGDemo: /usr/local/lib/libopencv_objdetect.so.2.4.9
TGDemo: /usr/local/lib/libopencv_legacy.so.2.4.9
TGDemo: /usr/local/lib/libopencv_video.so.2.4.9
TGDemo: /usr/local/lib/libopencv_ml.so.2.4.9
TGDemo: /usr/local/lib/libopencv_calib3d.so.2.4.9
TGDemo: /usr/local/lib/libopencv_features2d.so.2.4.9
TGDemo: /usr/local/lib/libopencv_highgui.so.2.4.9
TGDemo: /usr/local/lib/libopencv_imgproc.so.2.4.9
TGDemo: /usr/local/lib/libopencv_flann.so.2.4.9
TGDemo: /usr/local/lib/libopencv_core.so.2.4.9
TGDemo: CMakeFiles/TGDemo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiaming/object_tracking_2D/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable TGDemo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TGDemo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TGDemo.dir/build: TGDemo

.PHONY : CMakeFiles/TGDemo.dir/build

CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/include/object_tracking_2D/utils.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/templates_generator.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/Camera.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/EdgeTracker.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/epnp.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/glm.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/HomoTransform.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/ModelImport.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/objectmodel.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/ParticleFilter.cpp.o.requires
CMakeFiles/TGDemo.dir/requires: CMakeFiles/TGDemo.dir/src/PoseEstimationSURF.cpp.o.requires

.PHONY : CMakeFiles/TGDemo.dir/requires

CMakeFiles/TGDemo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TGDemo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TGDemo.dir/clean

CMakeFiles/TGDemo.dir/depend:
	cd /home/jiaming/object_tracking_2D/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiaming/object_tracking_2D /home/jiaming/object_tracking_2D /home/jiaming/object_tracking_2D/build /home/jiaming/object_tracking_2D/build /home/jiaming/object_tracking_2D/build/CMakeFiles/TGDemo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TGDemo.dir/depend

