#include "plugin/ReshapePlugin.h"
#include "cudaUtility.h"

ReshapePlugin::ReshapePlugin(int outC) 
    : mOutC(outC)
{ }

ReshapePlugin::ReshapePlugin(const void *buffer, size_t size, int outC)
    : mOutC(outC)
{
    CHECK_EQ(size, sizeof(mCopySize));
    mCopySize = *reinterpret_cast<const size_t *>(buffer);
}

int ReshapePlugin::getNbOutputs() const
{ 
    return 1; 
}

nvinfer1::Dims ReshapePlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims)
{
    CHECK_EQ(nbInputDims, 1);
    CHECK_EQ(index, 0);
    CHECK_EQ(inputs[index].nbDims, 3);
    CHECK_EQ((inputs[0].d[0]) * (inputs[0].d[1]) % mOutC, 0);

    return nvinfer1::DimsCHW(mOutC, inputs[0].d[0] * inputs[0].d[1] / mOutC, inputs[0].d[2]);
}

int ReshapePlugin::initialize()
{
    return 0;
}

void ReshapePlugin::terminate()
{ }

size_t ReshapePlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
int ReshapePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    cudaError_t status = cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream);

    CUDA_CHECK(status);

    return 0;
}

size_t ReshapePlugin::getSerializationSize()
{
    return sizeof(mCopySize);
}

void ReshapePlugin::serialize(void *buffer)
{
    *reinterpret_cast<size_t *>(buffer) = mCopySize;
}

void ReshapePlugin::configure(const nvinfer1::Dims *inputs, int nbInputs, const nvinfer1::Dims *outputs, int nbOutputs, int maxBatchSize)
{
    mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);

#ifdef _DEBUG
    LOG(INFO) << "reshape layer(" << inputs[0].d[0] << ", " << inputs[0].d[1] << ", " << inputs[0].d[2] << ")";
    LOG(INFO) << "copy size: " << mCopySize;
#endif
}