#include "plugin/FlattenPlugin.h"
#include "cudaUtility.h"

FlattenPlugin::FlattenPlugin()
{ }

FlattenPlugin::FlattenPlugin(const void *buffer, size_t size)
{
    CHECK_EQ(size, 3 * sizeof(int));
    const int *d = reinterpret_cast<const int *>(buffer);
    _size = d[0] * d[1] * d[2];
    dimBottom = nvinfer1::DimsCHW{d[0], d[1], d[2]};
}

int FlattenPlugin::getNbOutputs() const
{ 
    return 1; 
}

nvinfer1::Dims FlattenPlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims)
{
    CHECK_EQ(nbInputDims, 1);
    CHECK_EQ(index, 0);
    CHECK_EQ(inputs[index].nbDims, 3);

    _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
    return nvinfer1::DimsCHW(_size, 1, 1);
}

int FlattenPlugin::initialize()
{
    return 0;
}
void FlattenPlugin::terminate()
{ }

size_t FlattenPlugin::getWorkspaceSize(int maxBatchSize) const
{ 
    return 0; 
}

int FlattenPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    cudaError_t status = cudaMemcpyAsync(outputs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    CUDA_CHECK(status);

    return 0;
}

size_t FlattenPlugin::getSerializationSize()
{
    return 3 * sizeof(int);
}

void FlattenPlugin::serialize(void* buffer)
{
    int* d = reinterpret_cast<int*>(buffer);

    d[0] = dimBottom.c();
    d[1] = dimBottom.h();
    d[2] = dimBottom.w();
}

void FlattenPlugin::configure(const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize)
{
    dimBottom = nvinfer1::DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}