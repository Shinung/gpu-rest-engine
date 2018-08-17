#include "plugin/SoftmaxPlugin.h"
#include "cudaUtility.h"

using namespace trt::cudnn;

int SoftmaxPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    //LOG(INFO) << "bottom_desc_ set";
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc_, dataType<float>::type, 
                                            mBottomH, mBottomC, 1, 1,
                                            mBottomC, 1, 1, 1));
    //LOG(INFO) << "top_desc_ set";
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc_, dataType<float>::type, 
                                            mBottomH, mBottomC, 1, 1,
                                            mBottomC, 1, 1, 1));
    
    CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        dataType<float>::one,
        bottom_desc_, *inputs,
        dataType<float>::zero,
        top_desc_, *outputs));
    
    return 0;
}