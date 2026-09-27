# 3dgs/CMakeLists.txt
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wno-error=cpp")


cmake_minimum_required(VERSION 3.24)
project(3dgs LANGUAGES CUDA CXX)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86")
# Determine the project root directory
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Define the absolute path to libtorch


set(SIMPLE_KNN_DIR "/usr/local" CACHE PATH "simple-knn install prefix")
include_directories(${SIMPLE_KNN_DIR}/include/simple-knn/)
link_directories(${SIMPLE_KNN_DIR}/lib)

# Set CMAKE_PREFIX_PATH


# Core count calculation
include(ProcessorCount)
ProcessorCount(total_cores)
if (NOT DEFINED total_cores OR total_cores EQUAL 0)
    set(total_cores 1)
    set(used_cores 1)
elseif (total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2") # use total_cores-2 if total_cores > 1
endif ()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building with ${used_cores} out of ${total_cores} available cores")

# Files
set(HEADERS
        includes/read_utils.cuh
        includes/gaussian.cuh
        includes/camera.cuh
        includes/loss_monitor.cuh
        includes/image.cuh
        includes/stb_image.h
        includes/stb_image_resize.h
        includes/point_cloud.cuh
        includes/scene.cuh
        includes/scene_info.cuh
        includes/camera_utils.cuh
        includes/general_utils.cuh
        includes/sh_utils.cuh
        includes/parameters.cuh
        includes/loss_utils.cuh
        includes/render_utils.cuh
        includes/rasterizer.cuh
        includes/rasterize_points.cuh
        includes/camera_info.cuh
        includes/debug_utils.cuh
        cuda_rasterizer/backward.h
        cuda_rasterizer/forward.h
        cuda_rasterizer/auxiliary.h
        cuda_rasterizer/rasterizer_impl.h
        cuda_rasterizer/rasterizer.h
        cuda_rasterizer/serialization.h
)

set(SOURCES
        # src/main.cu
        src/read_utils.cu
        src/gaussian.cu
        src/camera.cu
        src/loss_monitor.cu
        src/image.cu
        src/scene.cu
        src/camera_utils.cu
        src/general_utils.cu
        src/parameters.cu
        src/rasterizer.cu
        src/rasterize_points.cu
        cuda_rasterizer/backward.cu
        cuda_rasterizer/forward.cu
        cuda_rasterizer/rasterizer_impl.cu
)


# Get rid of pesky warnings from stb_image.h produced in src/utils.cu
# set_source_files_properties(src/utils.cu PROPERTIES COMPILE_FLAGS "-Xcudafe --diag_suppress=550 ")

# Project target
# add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

add_library(3dgs_lib SHARED ${SOURCES} ${HEADERS})

# add_executable(${PROJECT_NAME}  src/main.cu)

# target_link_libraries(${PROJECT_NAME} PRIVATE 3dgs_lib ${TORCH_LIBRARIES} )

set_target_properties(3dgs_lib PROPERTIES CUDA_ARCHITECTURES native CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)


target_include_directories(3dgs_lib PRIVATE ${PROJECT_SOURCE_DIR}/includes ${PROJECT_SOURCE_DIR}/cuda_rasterizer ${PROJECT_SOURCE_DIR}/external/tinyply/source)

# target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/includes ${PROJECT_SOURCE_DIR}/cuda_rasterizer ${PROJECT_SOURCE_DIR}/external/tinyply/source)

# Torch setup
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-error=cpp" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

# Check CUDA version
if (CUDAToolkit_VERSION VERSION_LESS "11.7")
    message(FATAL_ERROR "This project requires CUDA 11.7 or higher")
endif ()

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch" CACHE PATH "LibTorch root")
include_directories(${LIBTORCH_DIR}/include)
link_directories(${LIBTORCH_DIR}/lib)

# set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR};${CMAKE_PREFIX_PATH}")
# find_package(Torch REQUIRED)
find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR} NO_DEFAULT_PATH)


find_package(CUDA REQUIRED)

find_package(tinyply REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(nlohmann_json REQUIRED)
find_package(glm REQUIRED)
find_package(Threads REQUIRED)

find_package(yaml-cpp REQUIRED)
target_link_libraries(3dgs_lib
        PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${OpenCV_LIBS}
        simple-knn
        tinyply
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        glm::glm
        Threads::Threads
        yaml-cpp::yaml-cpp
)



install(TARGETS 3dgs_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/includes/ DESTINATION include)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/cuda_rasterizer/ DESTINATION include)