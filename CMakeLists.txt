cmake_minimum_required(VERSION 3.8)
project(indoor_search)

# 启用 CMP0074：find_package 使用 <PackageName>_ROOT 变量（消除 PCL_ROOT 警告）
cmake_policy(SET CMP0074 NEW)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(std_srvs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(px4_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(msg_files
  msg/SearchPath.msg
  msg/LocalCommand.msg
  msg/ApfCommand.msg
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs
)

add_executable(octomap_builder src/octomap_builder.cpp)
target_link_libraries(octomap_builder ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
ament_target_dependencies(octomap_builder rclcpp octomap_msgs nav_msgs geometry_msgs sensor_msgs px4_msgs pcl_conversions PCL octomap std_srvs)

add_executable(rrt_star_3d_planner src/rrt_star_3d_planner.cpp)
ament_target_dependencies(rrt_star_3d_planner rclcpp geometry_msgs nav_msgs octomap octomap_msgs tf2 tf2_ros)

add_executable(offboard_control src/offboard_control.cpp)
ament_target_dependencies(offboard_control rclcpp px4_msgs geometry_msgs nav_msgs tf2 tf2_ros)

install(TARGETS
  octomap_builder
  rrt_star_3d_planner
  offboard_control
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  src/red_cube_detector.py
  src/target_search_node.py
  src/px4_slam_bridge_node.py
  src/target_search_constants.py
  src/target_search_types.py
  src/safety_monitor.py
  src/target_confirmation.py
  src/global_planner_node.py
  src/local_mpc_node.py
  src/apf_node.py
  src/failsafe_manager.py
  src/config_manager.py
  src/grid_utils.py
  scripts/sim_test_scenarios.py
  scripts/system_diagnostics.py
  scripts/offboard_commander.py
  DESTINATION lib/${PROJECT_NAME})

# Register python executables so `ros2 run indoor_search <name>` works
set(_indoor_search_py_execs
  apf_node.py
  global_planner_node.py
  local_mpc_node.py
  target_search_node.py
  px4_slam_bridge_node.py
  red_cube_detector.py
  failsafe_manager.py
  config_manager.py
  grid_utils.py
  sim_test_scenarios.py
  system_diagnostics.py
  offboard_commander.py
)
string(REPLACE ";" "\n" _indoor_search_exec_lines "${_indoor_search_py_execs}")
set(_ros2run_registry "${CMAKE_CURRENT_BINARY_DIR}/ros2run_registry")
file(WRITE "${_ros2run_registry}" "${_indoor_search_exec_lines}\n")
install(FILES "${_ros2run_registry}"
  DESTINATION share/ament_index/resource_index/ros2run
  RENAME ${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)

ament_package()
