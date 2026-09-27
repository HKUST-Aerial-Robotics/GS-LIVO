cmake_minimum_required(VERSION 3.18)
project(fast_livo CUDA CXX)

# External prefixes (see README "Build"):
#   GSLIVO_DEPS_DIR  install prefix of lib3dgs, simple-knn and tinyply (src/lib3dgs, src/simple-knn, src/tinyply)
#   LIBTORCH_DIR     LibTorch (cxx11 ABI, CUDA build)
#   GSVIZ_CPP_DIR    optional: gsviz/slam/cpp of the gsviz viewer -> live streaming to the viewer
set(GSLIVO_DEPS_DIR "/usr/local" CACHE PATH "install prefix of lib3dgs / simple-knn / tinyply")
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch" CACHE PATH "LibTorch root")
set(GSVIZ_CPP_DIR "" CACHE PATH "gsviz/slam/cpp (optional viewer bridge)")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops")
  add_definitions(-DX86_ARCH)
endif()

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

include(ProcessorCount)
ProcessorCount(N)
if(N GREATER 4)
  add_definitions(-DMP_EN -DMP_PROC_NUM=4)
elseif(N GREATER 1)
  add_definitions(-DMP_EN -DMP_PROC_NUM=${N})
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# LibTorch. Its static libkineto.a needs CUPTI, which a runtime-only CUDA install lacks;
# libtorch_cpu.so carries its own copy, so it is not needed for linking.
set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "kineto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
endif()
find_package(mimalloc QUIET)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs nav_msgs sensor_msgs roscpp rospy std_msgs pcl_ros tf message_generation
  eigen_conversions vikit_common vikit_ros cv_bridge image_transport
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(PythonLibs QUIET)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${GSLIVO_DEPS_DIR}/include
  ${GSLIVO_DEPS_DIR}/include/simple-knn
)
link_directories(${GSLIVO_DEPS_DIR}/lib)
set(LIB3DGS_LIBRARY ${GSLIVO_DEPS_DIR}/lib/lib3dgs_lib.so)

if(GSVIZ_CPP_DIR)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JSONCPP REQUIRED jsoncpp)
  include_directories(${GSVIZ_CPP_DIR} ${JSONCPP_INCLUDE_DIRS})
  link_directories(${JSONCPP_LIBRARY_DIRS})
  add_definitions(-DWITH_GSVIZ)
  message(STATUS "gsviz viewer bridge: ${GSVIZ_CPP_DIR}")
endif()

# Gaussian map (hash-indexed octree + sliding window, Sec. II-A..C)
add_library(gs_map src/gs_map.cpp)
target_link_libraries(gs_map ${LIB3DGS_LIBRARY} ${TORCH_LIBRARIES})

add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
target_link_libraries(vio gs_map ${LIB3DGS_LIBRARY} ${JSONCPP_LIBRARIES})
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)

add_executable(fastlivo_mapping src/main.cpp)
target_link_libraries(fastlivo_mapping
  laser_mapping vio gs_map lio pre imu_proc
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES}
  ${TORCH_LIBRARIES} ${JSONCPP_LIBRARIES}
)
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()
