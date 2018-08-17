#ifndef __CUDA_UTILITY_H_
#define __CUDA_UTILITY_H_

#include <cuda_runtime.h>
#include <cudnn.h>
#include <glog/logging.h>

//
// CUDA macros
//
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDNN_CHECK(condition) \
    do { \
        cudnnStatus_t status = condition; \
        CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " " \
            << cudnnGetErrorString(status); \
    } while(0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace trt {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float> {
  public:
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
    static float oneval, zeroval;
    static const void *one, *zero;
};
template<> class dataType<double> {
  public:
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    static double oneval, zeroval;
    static const void *one, *zero;
};

} // namespace cudnn
} // namespace trt

#endif
