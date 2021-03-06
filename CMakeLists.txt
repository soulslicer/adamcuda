cmake_minimum_required (VERSION 2.8)
project(hand_model)

SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
SET(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -Wextra")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
find_package( OpenCV 2.4 REQUIRED PATHS "/home/donglaix/tools/opencv-2.4.13/build/")
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)
include_directories(${CERES_INCLUDE_DIRS})
set(IGL_INCLUDE_DIRS "/home/raaj/libigl/include/")
include_directories(${IGL_INCLUDE_DIRS})
include_directories("include")
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
#find_package(GFlags REQUIRED)

######## Lib for CUDA
file(GLOB ADAMCUDA_SOURCES "src/*.cu")
set(CMAKE_PREFIX_PATH "/usr/local/lib/python2.7/dist-packages/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)
#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})
find_package(CUDA REQUIRED)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS}
${TORCH_INCLUDE_DIRS}
)
#cuda_add_library(adamcuda ${ADAMCUDA_SOURCES})
#target_link_libraries( adamcuda ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
#set_property(TARGET adamcuda PROPERTY CXX_STANDARD 11)
##########

add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")


#find_package(PkgConfig REQUIRED)
#pkg_check_modules(JSONCPP jsoncpp)
#link_libraries(${JSONCPP_LIBRARIES})

set(3rdparty_libraries -ljsoncpp ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY})

set(CUDA_NVCC_FLAGS -std=c++11 -g -Xcompiler -fexceptions -Xcompiler -fPIC -w)
file(GLOB SOURCES "src/*.cpp")
cuda_add_library(MYLIB ${SOURCES} ${ADAMCUDA_SOURCES})
target_link_libraries( MYLIB ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} )

add_executable(main main.cpp)
target_link_libraries( main MYLIB )
target_link_libraries( main ${3rdparty_libraries} )

add_subdirectory(3rdparty/pybind11)
add_subdirectory(python)
