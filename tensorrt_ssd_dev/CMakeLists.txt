cmake_minimum_required(VERSION 3.8)
project(ssdEngine CUDA CXX DESCRIPTION "ssdEngine optimized by TensorRT")

FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(OPENCV REQUIRED opencv)
PKG_CHECK_MODULES(CUDART REQUIRED cudart-9.0)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_61,code=sm_61
)

add_library(ssdEngine SHARED
    src/caffe.pb.cc
    src/gpu_allocator.cpp
    src/InferenceEngine.cpp
    src/detection.cpp
    src/pluginImplement.cpp
    src/SSD.cpp
    src/plugin/FlattenPlugin.cpp
    src/plugin/ReshapePlugin.cpp
    src/plugin/SoftmaxPlugin.cpp
    src/cudaUtility.cpp
    Cuda/SoftmaxPlugin.cu
)

set_target_properties(ssdEngine PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    CUDA_SEPARABLE_COMPILATION ON
    PUBLIC_HEADER LPR.h)

target_include_directories(ssdEngine PRIVATE 
    ${OPENCV_INCLUDE_DIRS}
    ${CUDART_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/..
    /opt/TensorRT-4.0.0.3/include)

target_link_libraries(ssdEngine
    ${OPENCV_LIBRARIES}
    ${CUDART_LIBRARIES}
    /opt/TensorRT-4.0.0.3/lib/libnvinfer.so
    /opt/TensorRT-4.0.0.3/lib/libnvinfer_plugin.so
    /opt/TensorRT-4.0.0.3/lib/libnvcaffe_parser.so
    /opt/TensorRT-4.0.0.3/lib/libnvparsers.so
    cudnn protobuf glog boost_system boost_thread)

message(${OPENCV_INCLUDE_DIRS})
message(${CUDART_INCLUDE_DIRS})
message(${CUDART_LIBRARIES})
message(${CUDART_LIBRARY_DIRS})

link_directories(${CUDART_LIBRARY_DIRS})

add_executable(ssdEngineDebug 
                src/main.cpp)

target_include_directories(ssdEngineDebug PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(ssdEngineDebug
    ssdEngine)