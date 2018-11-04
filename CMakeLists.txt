cmake_minimum_required(VERSION 2.8.3)

project(uav_tracking)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

add_subdirectory(src/inc/fhog)

find_package(catkin REQUIRED COMPONENTS
	message_generation
        roscpp
        geometry_msgs
        dji_sdk
        sensor_msgs
	cv_bridge
	std_msgs
	image_transport
)

find_package(OpenCV 2.4.13 REQUIRED)
message(${OpenCV_VERSION})
#find_package(OpenCV 2.4.13 REQUIRED)
set(CMAKE_MODULE_PATH /usr/local/lib/cmake)
find_package(Boost REQUIRED COMPONENTS system)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${CMAKE_CXX_FLAGS}")
#AUX_SOURCE_DIRECTORY(src/inc2 SRC)
add_message_files(
  FILES
	boundingbox.msg
)
add_service_files(
  FILES
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  LIBRARIES dji_sdk
  CATKIN_DEPENDS roscpp std_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

add_executable(main
	src/main.cpp #${SRC}
)

target_link_libraries(main
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	fhog
	
)

add_executable(bbox_pub
	src/bbox_pub.cpp
)
target_link_libraries(bbox_pub
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
)


