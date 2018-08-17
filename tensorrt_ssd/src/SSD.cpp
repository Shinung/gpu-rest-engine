#include "SSD.h"

#include <iosfwd>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#if defined(_MSC_VER)
#include <io.h>
#endif
#include <fcntl.h>

#include "NvCaffeParser.h"
#include "caffe.pb.h"

using namespace caffe;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;

SSD::SSD(std::shared_ptr<InferenceEngine> engine,
		const std::string& mean_file,
		const std::string& mean_value,
        const string& label_file,
        GPUAllocator* allocator)
    : allocator_(allocator),
      engine_(engine)
{
    SetModel();
    SetMean(mean_file, mean_value);
    SetLabels(label_file);
}

SSD::~SSD()
{
	context_->destroy();
	CHECK_EQ(cudaFree(input_layer_), cudaSuccess) << "Could not free input layer";
	CHECK_EQ(cudaFree(output_layer_), cudaSuccess) << "Could not free output layer";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector< std::vector<float> >& v, int N)
{
    std::vector< std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i][2], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
DetectedBBoxes SSD::Detect(const Mat& img, int N)
{
	Size frameSize(img.cols, img.rows);
    std::vector< std::vector<float> > output = Predict(img);

	N = N <= 0 ? output.size() : std::min<int>(output.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    DetectedBBoxes detectedBoxes;
    for (int i = 0; i < N; ++i)
    {
        int idx = maxN[i];
		detectedBoxes.emplace_back(output[idx], labels_[cvRound(output[idx][1])], frameSize);
    }

    return detectedBoxes;
}

std::vector< std::vector<float> > SSD::Predict(const Mat& img)
{
    std::vector<GpuMat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    void* buffers[2] = { input_layer_, output_layer_ };
    context_->execute(1, buffers);

    size_t output_size = output_dim_.c() * output_dim_.h() * output_dim_.w();
    std::vector<float> output(output_size);
    cudaError_t st = cudaMemcpy(output.data(), output_layer_, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess)
        throw std::runtime_error("could not copy output layer back to host");

    auto iter = output.cbegin();
    std::vector< std::vector<float> > detections;
    while(iter != output.cend())
    {
        if(*(iter + 1) == -1)
        {
            iter += 7;
            continue;
        }
        std::vector<float> detection(iter, iter + 7);
        detections.push_back(detection);
        iter += 7;
    }
    return detections;
}

void SSD::SetModel()
{
    ICudaEngine* engine = engine_->Get();

    context_ = engine->createExecutionContext();
    CHECK(context_) << "Failed to create execution context.";

    int input_index = engine->getBindingIndex("data");
    input_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(input_index));
    input_cv_size_ = Size(input_dim_.w(), input_dim_.h());
    // FIXME: could be wrapped in a thrust or GpuMat object.
    size_t input_size = input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);
    cudaError_t st = cudaMalloc(&input_layer_, input_size);
    CHECK_EQ(st, cudaSuccess) << "Could not allocate input layer.";

    int output_index = engine->getBindingIndex("detection_out");
    output_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(output_index));
    size_t output_size = output_dim_.c() * output_dim_.h() * output_dim_.w() * sizeof(float);
    st = cudaMalloc(&output_layer_, output_size);
    CHECK_EQ(st, cudaSuccess) << "Could not allocate output layer.";
}

/* Load the mean file in binaryproto format. */
void SSD::SetMean(const string& mean_file, const string& mean_value)
{
    if (!mean_file.empty())
    {
        ICaffeParser *parser = createCaffeParser();
        IBinaryProtoBlob *mean_blob = parser->parseBinaryProto(mean_file.c_str());
        parser->destroy();
        CHECK(mean_blob) << "Could not load mean file.";

        DimsNCHW mean_dim = mean_blob->getDimensions();
        int c = mean_dim.c();
        int h = mean_dim.h();
        int w = mean_dim.w();
        CHECK_EQ(c, input_dim_.c())
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<Mat> channels;
        float *data = (float *)mean_blob->getData();
        for (int i = 0; i < c; ++i)
        {
            /* Extract an individual channel. */
            Mat channel(h, w, CV_32FC1, data);
            channels.push_back(channel);
            data += h * w;
        }

        /* Merge the separate channels into a single image. */
        Mat packed_mean;
        merge(channels, packed_mean);

        /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
        Scalar channel_mean = mean(packed_mean);
        Mat host_mean = Mat(input_cv_size_, packed_mean.type(), channel_mean);
        mean_.upload(host_mean);
    }
    if (!mean_value.empty())
    {
        CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
        std::stringstream ss(mean_value);
        std::vector<float> values;
        string item;
        int num_channels_ = input_dim_.c();
        
        while (getline(ss, item, ','))
        {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == static_cast<size_t>(num_channels_))
            << "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            Mat channel(input_dim_.h(), input_dim_.w(), CV_32FC1, Scalar(values[i]));
            channels.push_back(channel);
        }

        Mat packed_mean;
        merge(channels, packed_mean);

        mean_.upload(packed_mean);
    }
}

void SSD::SetLabels(const string& label_file)
{
    /* Load labels. */
	LabelMap labelMap;
#if defined (_MSC_VER)  // for MSC compiler binary flag needs to be specified
	int fd;
	_sopen_s(&fd, label_file.c_str(), O_RDONLY | O_BINARY, _SH_DENYNO, 0);
#else
	int fd = open(label_file.c_str(), O_RDONLY);
#endif
	CHECK_NE(fd, -1) << "Unable to open labels file " << label_file;
	google::protobuf::io::FileInputStream fileInput(fd);
	fileInput.SetCloseOnDelete(true);
	CHECK(google::protobuf::TextFormat::Parse(&fileInput, &labelMap)) << "Parse failed from: " << label_file;
	
	for (int i = 0; i < labelMap.item_size(); ++i)
	{
		const LabelMapItem& item = labelMap.item(i);
		labels_.push_back(item.display_name());
	}

	////////////////////////////////
	//Retrive labels
	////////////////////////////////
	/*if (!labels_.empty())
	{
		std::ofstream ostream("labels.txt");
		CHECK(ostream.is_open()) << "labels.txt isn't open";
		for (size_t i = 0; i < labels_.size(); ++i)
		{
			ostream << i << " " << labels_[i] << std::endl;
		}
		ostream.close();
	}*/

    // CHECK_EQ(labels_.size(), net_->layer_by_name("detection_out")->layer_param().detection_output_param().num_classes())
    //     << "Number of labels is different from the output layer's parameter.";
}

/* Wrap the input layer of the network in separate GpuMat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SSD::WrapInputLayer(std::vector<GpuMat>* input_channels)
{
    int width = input_dim_.w();
    int height = input_dim_.h();
    float* input_data = input_layer_;
    for (int i = 0; i < input_dim_.c(); ++i)
    {
        GpuMat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void SSD::Preprocess(const Mat& host_img,
                            std::vector<GpuMat>* input_channels)
{
    int num_channels_ = input_dim_.c();
    GpuMat img(host_img, allocator_);
    /* Convert the input image to the input image format of the network. */
    GpuMat sample(allocator_);
    if (img.channels() == 3 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    GpuMat sample_resized(allocator_);
    if (sample.size() != input_cv_size_)
        cuda::resize(sample, sample_resized, input_cv_size_);
    else
        sample_resized = sample;

    GpuMat sample_float(allocator_);
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    GpuMat sample_normalized(allocator_);
    cuda::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the GpuMat
     * objects in input_channels. */
    cuda::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == input_layer_)
        << "Input channels are not wrapping the input layer of the network.";
}