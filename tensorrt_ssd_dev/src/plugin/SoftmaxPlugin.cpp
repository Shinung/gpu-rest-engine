#include "plugin/SoftmaxPlugin.h"
#include "cudaUtility.h"

using namespace trt::cudnn;

SoftmaxPlugin::SoftmaxPlugin()
    : handles_setup_(false),
      handle_(nullptr),
      bottom_desc_(nullptr),
      top_desc_(nullptr),
      mCopySize(0),
      mBottomC(0),
      mBottomH(0),
      mBottomW(0)
{ }

SoftmaxPlugin::SoftmaxPlugin(const void *buffer, size_t size)
    : handles_setup_(false),
      handle_(nullptr),
      bottom_desc_(nullptr),
      top_desc_(nullptr),
      mCopySize(0),
      mBottomC(0),
      mBottomH(0),
      mBottomW(0)
{
    CHECK_EQ(size, sizeof(mCopySize));
    mCopySize = *reinterpret_cast<const size_t *>(buffer);
}

int SoftmaxPlugin::getNbOutputs() const
{ 
    return 1; 
}

nvinfer1::Dims SoftmaxPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    CHECK_EQ(nbInputDims, 1);
    CHECK_EQ(index, 0);
    CHECK_EQ(inputs[index].nbDims, 3);

    return nvinfer1::DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

void SoftmaxPlugin::configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize)
{
    /* In Caffe SSD
    softmax_axis = 2: this is declared in deploy.prototxt

    N = this->outer_num_ = num * channels
    K = shape(softmax_axis_) = height
    H = this->inner_num_ = width
    W = 1
    */
#ifdef _DEBUG
    LOG(INFO) << outputDims[0].d[0] << " " << outputDims[0].d[1] << " " << outputDims[0].d[2];
#endif

    mBottomC = inputDims[0].d[0];
    mBottomH = inputDims[0].d[1];
    mBottomW = inputDims[0].d[2];

    mCopySize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2] * sizeof(float);
}

int SoftmaxPlugin::initialize()
{
#ifdef _DEBUG
    LOG(INFO) << "SoftmaxPlugin initialize";
#endif

    CUDNN_CHECK(cudnnCreate(&handle_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));

    handles_setup_ = true;

    return 0;
}

void SoftmaxPlugin::terminate()
{
    if (!handles_setup_)
    {
        return;
    }

    cudnnDestroyTensorDescriptor(bottom_desc_);
    cudnnDestroyTensorDescriptor(top_desc_);
    cudnnDestroy(handle_);
}

size_t SoftmaxPlugin::getWorkspaceSize(int maxBatchSize) const
{ 
    return 0; 
}

size_t SoftmaxPlugin::getSerializationSize()
{
    return sizeof(mCopySize);
}

void SoftmaxPlugin::serialize(void* buffer)
{
    *reinterpret_cast<size_t*>(buffer) = mCopySize;
}