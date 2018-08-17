#ifndef RESHAPE_PLUGIN_H
#define RESHAPE_PLUGIN_H

#include "NvInferPlugin.h"

//SSD Reshape layer : shape{0,-1,21}
class ReshapePlugin : public nvinfer1::IPlugin
{
  public:
    explicit ReshapePlugin(int outC);
    ReshapePlugin(const void *buffer, size_t size, int outC);

    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const nvinfer1::Dims *inputs, int nbInputs, const nvinfer1::Dims *outputs, int nbOutputs, int maxBatchSize) override;

  protected:
    int mOutC;
    size_t mCopySize;
};

#endif