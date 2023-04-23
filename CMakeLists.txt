cmake_minimum_required(VERSION 3.10)

project(yolov5_obb)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)



include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)
# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g  -D_MWAITXINTRIN_H_INCLUDED")#-Wfatal-errors
cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)


cuda_add_executable(yolov5_gen  yolov5_gen.cpp)

target_link_libraries(yolov5_gen nvinfer)
target_link_libraries(yolov5_gen cudart)
target_link_libraries(yolov5_gen myplugins)
target_link_libraries(yolov5_gen ${OpenCV_LIBS})

cuda_add_executable(yolov5_use yolov5_use.cpp)

target_link_libraries(yolov5_use nvinfer)
target_link_libraries(yolov5_use cudart)
target_link_libraries(yolov5_use myplugins)
target_link_libraries(yolov5_use ${OpenCV_LIBS})


if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


