#ifndef SOFTMAX_PLUGIN_H
#define SOFTMAX_PLUGIN_H

#include "NvInferPlugin.h"

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

struct cudnnTensorStruct;
typedef struct cudnnTensorStruct*          cudnnTensorDescriptor_t;

//Softmax layer . TensorRT softmax only support cross channel
class SoftmaxPlugin : public nvinfer1::IPlugin
{
    //You need to implement it when softmax parameter axis is 2.
  public:
    SoftmaxPlugin();
    SoftmaxPlugin(const void *buffer, size_t size);

    int getNbOutputs() const override;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    void configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize) override;
    int initialize() override;
    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

  protected:
    bool handles_setup_;
    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t bottom_desc_;
    cudnnTensorDescriptor_t top_desc_;
    size_t mCopySize;

    int mBottomC, mBottomH, mBottomW;
};

#endif