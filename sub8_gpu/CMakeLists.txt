cmake_minimum_required(VERSION 2.8.3)
project(sub8_gpu)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
  image_geometry
  message_filters
)

find_package( OpenCV 2.4.11 REQUIRED )
find_package( PCL )

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
  INCLUDE_DIRS 
# LIBRARIES sub8_gpu_stereoCM_node
  CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge
  DEPENDS system_lib cv_bridge
)

#add_subdirectory(include/backward_cpp)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(sub8_gpu_stereo src/sub8_gpu_stereo.cpp)

# Sadly backward-cpp is not compatible with ARM-HF
#add_executable(sub8_gpu_stereoCM_node_node src/sub8_gpu_stereoCM_node.cpp ${backward_ENABLE})

#add_backward(sub8_gpu_stereoCM_node_node)

## Specify libraries to link a library or executable target against
message(${catkin_LIBRARIES})
target_link_libraries(sub8_gpu_stereo
   ${PCL_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )
