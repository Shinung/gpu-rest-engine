#include "detection.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "SSD.h"

using namespace caffe;
using std::string;

/* By using Go as the HTTP server, we have potentially more CPU threads than
* available GPUs and more threads can be added on the fly by the Go
* runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
* when a CPU thread is ready for inference it will try to retrieve an
* execution context from a queue of available GPU contexts and then do a
* cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
* per GPU. */
class ExecContext
{
public:
	friend ScopedContext<ExecContext>;

	static bool IsCompatible(int device)
	{
		cudaError_t st = cudaSetDevice(device);
		if (st != cudaSuccess)
			return false;

		cuda::DeviceInfo info;
		if (!info.isCompatible())
			return false;

		return true;
	}

	ExecContext(const string& model_file,
				const string& trained_file,
				const string& mean_file,
				const string& mean_value,
				const string& label_file,
				int device)
		: device_(device)
	{
		cudaError_t st = cudaSetDevice(device_);
		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");

		allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
		caffe_context_.reset(new Caffe);
		Caffe::Set(caffe_context_.get());

		detector_.reset(new SSD(model_file, trained_file,
				mean_file, mean_value,
				label_file,
				allocator_.get()));

		Caffe::Set(nullptr);
	}

	SSD* CaffeDetector()
	{
		return detector_.get();
	}

private:
	void Activate()
	{
		cudaError_t st = cudaSetDevice(device_);
		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");
		allocator_->reset();
		Caffe::Set(caffe_context_.get());
	}

	void Deactivate()
	{
		Caffe::Set(nullptr);
	}

private:
	int device_;
	std::unique_ptr<GPUAllocator> allocator_;
	std::unique_ptr<Caffe> caffe_context_;
	std::unique_ptr<SSD> detector_;
};

struct detection_ctx
{
	ContextPool<ExecContext> pool;
};

/* Currently, 2 execution contexts are created per GPU. In other words, 2
* inference tasks can execute in parallel on the same GPU. This helps improve
* GPU utilization since some kernel operations of inference will not fully use
* the GPU. */
constexpr static int kContextsPerDevice = 2;

detection_ctx* detection_initialize(const char* model_file, const char* trained_file,
		const char* mean_file, const char* mean_value, const char* label_file)
{
	try
	{
		::google::InitGoogleLogging("inference_server");

		int device_count;
		cudaError_t st = cudaGetDeviceCount(&device_count);
		if (st != cudaSuccess)
			throw std::invalid_argument("could not list CUDA devices");

		ContextPool<ExecContext> pool;
		for (int dev = 0; dev < device_count; ++dev)
		{
			if (!ExecContext::IsCompatible(dev))
			{
				LOG(ERROR) << "Skipping device: " << dev;
				continue;
			}

			for (int i = 0; i < kContextsPerDevice; ++i)
			{
				std::unique_ptr<ExecContext> context(new ExecContext(model_file, trained_file,
					mean_file, mean_value, label_file, dev));
				pool.Push(std::move(context));
			}
		}

		if (pool.Size() == 0)
			throw std::invalid_argument("no suitable CUDA device");

		detection_ctx* ctx = new detection_ctx{ std::move(pool) };

		/* Successful CUDA calls can set errno. */
		errno = 0;
		return ctx;
	}
	catch (const std::invalid_argument& ex)
	{
		LOG(ERROR) << "exception: " << ex.what();
		errno = EINVAL;
		return nullptr;
	}
}

const char* detection_inference(detection_ctx* ctx, char* buffer, size_t length)
{
	try
	{
		_InputArray array(buffer, length);

		Mat img = imdecode(array, -1);
		if (img.empty())
			throw std::invalid_argument("could not decode image");

		time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		std::ostringstream strTime;
	    strTime << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");

		{
			std::ostringstream imgName;
			std::vector<int> jpeg_params;

			imgName << "./log/" << strTime.str() << ".jpg";
			jpeg_params.push_back(IMWRITE_JPEG_QUALITY);
			jpeg_params.push_back(80);
			
			imwrite(imgName.str(), img, jpeg_params);
		}

		Mat gray;
		if (img.channels() == 3)
		{
			cvtColor(img, gray, CV_BGR2GRAY);
		}
		else
		{
			gray = img;
		}

		DetectedBBoxes detections;
		std::ostringstream os;
		{
			/* In this scope an execution context is acquired for inference and it
			* will be automatically released back to the context pool when
			* exiting this scope. */
			ScopedContext<ExecContext> context(ctx->pool);
			SSD* detector = context->CaffeDetector();
			detections = detector->Detect(gray);
		}

		/* Write the top N predictions in JSON format. */
		os << "{\"num\":" << "\"" << detections.size() << "\",";
		os << "\"objects\":[";
		for (const BBox& box : detections)
		{
			os << "{";
			os << "\"confidence\":" << std::fixed << std::setprecision(4) << box.mConf << ",";
			os << "\"label\":" << "\"" << box.mLabel.second << "\",";
			os << "\"minx\":" << box.mBox.tl().x << ",";
			os << "\"miny\":" << box.mBox.tl().y << ",";
			os << "\"maxx\":" << box.mBox.br().x << ",";
			os << "\"maxy\":" << box.mBox.br().y << "},";
		}
		os.seekp(-1, std::ios_base::end);
		os << "]}";

		{
			std::ostringstream jsonName;
			jsonName << "./log/" << strTime.str() << ".json";

			std::ofstream outFile(jsonName.str());
			outFile << os.str();
			outFile.close();
		}

		errno = 0;
		std::string str = os.str();
#ifdef _MSC_VER
		return _strdup(str.c_str());
#else
		return strdup(str.c_str());
#endif
	}
	catch (const std::invalid_argument&)
	{
		errno = EINVAL;
		return nullptr;
	}
}

void detection_destroy(detection_ctx* ctx)
{
	delete ctx;
}
