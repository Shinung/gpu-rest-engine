#ifndef FLATTEN_PLUGIN_H
#define FLATTEN_PLUGIN_H

#include "NvInferPlugin.h"

//SSD Flatten layer
class FlattenPlugin : public nvinfer1::IPlugin
{
  public:
    FlattenPlugin();
    FlattenPlugin(const void *buffer, size_t size);

    int getNbOutputs() const override;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const nvinfer1::Dims *inputs, int nbInputs, const nvinfer1::Dims *outputs, int nbOutputs, int maxBatchSize) override;

  protected:
    nvinfer1::DimsCHW dimBottom;
    int _size;
};

#endif