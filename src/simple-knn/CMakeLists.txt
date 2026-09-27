# Plain CMake build of simple-knn as a shared library (lib3dgs links it; setup.py builds the Python extension).
cmake_minimum_required(VERSION 3.24)
project(simple-knn LANGUAGES CUDA CXX)
set(LIBTORCH_DIR "" CACHE PATH "LibTorch root")
find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR} NO_DEFAULT_PATH)
find_package(Python3 REQUIRED COMPONENTS Development)
add_library(simple-knn SHARED spatial.cu simple_knn.cu)
set_target_properties(simple-knn PROPERTIES CUDA_ARCHITECTURES native CUDA_STANDARD 17 CXX_STANDARD 17
                      POSITION_INDEPENDENT_CODE ON)
target_include_directories(simple-knn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${Python3_INCLUDE_DIRS})
target_link_libraries(simple-knn PRIVATE ${TORCH_LIBRARIES} Python3::Python)
install(TARGETS simple-knn LIBRARY DESTINATION lib)
install(FILES spatial.h simple_knn.h DESTINATION include/simple-knn)
