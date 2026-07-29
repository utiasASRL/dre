cmake_minimum_required(VERSION 3.8)
project(dre)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Match lgmath's compilation flags to avoid ABI issues
add_compile_options(-march=native)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)

# Build the vendored Dr-BA library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dre/ba ba_build)

install(TARGETS ba
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dre/ba/include/
  DESTINATION include
)

install(PROGRAMS
  dre/boreas_player.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME boreas_player
)

install(PROGRAMS
  dre/dro/dro_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dro_node
)

install(PROGRAMS
  dre/raplace/raplace_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME raplace_node
)

install(PROGRAMS
  dre/raplace/registration_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME registration_node
)

install(FILES
  dre/dro/dro.py
  dre/dro/dro_motion_models.py
  DESTINATION lib/${PROJECT_NAME}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadarInfo.msg"
  "msg/LocalMapInfo.msg"
  "msg/LoopCandidate.msg"
  "msg/DRLEstimate.msg"
   DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

add_executable(pogo_node
  dre/pogo/pogo_node.cpp dre/pogo/pose_graph.cpp
)
target_include_directories(pogo_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/dre/pogo
  ${YAML_CPP_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(pogo_node
  rclcpp
  nav_msgs
  geometry_msgs
)
target_link_libraries(pogo_node ${PROJECT_NAME}__rosidl_typesupport_cpp)

# Non-ament CMake packages must be linked explicitly.
if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(pogo_node yaml-cpp::yaml-cpp)
else()
  target_link_libraries(pogo_node yaml-cpp)
endif()

if(TARGET Ceres::ceres)
  target_link_libraries(pogo_node Ceres::ceres)
else()
  target_link_libraries(pogo_node ${CERES_LIBRARIES})
endif()

install(TARGETS pogo_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(mapping_node
  dre/ba/mapping_node.cpp
)
target_include_directories(mapping_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(mapping_node rclcpp sensor_msgs std_msgs nav_msgs message_filters)
target_link_libraries(mapping_node ${PROJECT_NAME}__rosidl_typesupport_cpp ba)

# Link remaining libraries
target_link_libraries(mapping_node
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(mapping_node yaml-cpp::yaml-cpp)
else()
  target_link_libraries(mapping_node yaml-cpp)
endif()

install(TARGETS mapping_node DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  dre/ba/map_viz_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME map_viz_node
)

add_executable(loc_node
  dre/drl/loc_node.cpp
)
target_include_directories(loc_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(loc_node rclcpp sensor_msgs std_msgs message_filters)
target_link_libraries(loc_node ${PROJECT_NAME}__rosidl_typesupport_cpp ba)

# Link remaining libraries
target_link_libraries(loc_node
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(loc_node yaml-cpp::yaml-cpp)
else()
  target_link_libraries(loc_node yaml-cpp)
endif()

install(TARGETS loc_node DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  dre/drl/loc_viz_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME loc_viz_node
)

install(PROGRAMS
  dre/drl/initial_pose_selector.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME initial_pose_selector
)


ament_package()
