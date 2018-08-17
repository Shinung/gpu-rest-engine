#ifndef SSD_H
#define SSD_H

#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <glog/logging.h>

#define USE_CUDNN 1
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "gpu_allocator.h"
#include "InferenceEngine.h"

class BBox
{
  public:
	BBox() {}
	BBox(const std::vector<float> &info, const std::string &label, const Size &frameSize)
	{
		mBox = Rect(Point(cvRound(info[3] * frameSize.width), cvRound(info[4] * frameSize.height)),
					Point(cvRound(info[5] * frameSize.width), cvRound(info[6] * frameSize.height)));

		mCenter = Point(cvRound((mBox.br().x + mBox.tl().x) / 2.f), cvRound((mBox.br().y + mBox.tl().y) / 2.f));

		mConf = info[2];

		mLabel = std::make_pair(cvRound(info[1]), label);
	}
	BBox(const BBox &other)
	{
		mBox = other.mBox;
		mCenter = other.mCenter;
		mConf = other.mConf;
		mLabel = other.mLabel;
	}
	BBox(const BBox &&rhs)
	{
		mBox = rhs.mBox;
		mCenter = rhs.mCenter;
		mConf = rhs.mConf;
		mLabel = rhs.mLabel;
	}

	BBox &operator=(const BBox &rhs)
	{
		if (this == &rhs)
		{
			return *this;
		}

		mBox = rhs.mBox;
		mCenter = rhs.mCenter;
		mConf = rhs.mConf;
		mLabel = rhs.mLabel;

		return *this;
	}

  public:
	Rect mBox;
	Point mCenter;
	float mConf;
	std::pair<int, std::string> mLabel;
};

typedef std::vector<BBox> DetectedBBoxes;

class SSD
{
  public:
	SSD(std::shared_ptr<InferenceEngine> engine,
		const std::string& mean_file,
		const std::string& mean_value,
		const std::string& label_file,
		GPUAllocator *allocator);

	~SSD();

	virtual DetectedBBoxes Detect(const Mat &img, int N = 5);

  protected:
	std::vector<std::vector<float>> Predict(const Mat &img);

  private:
	void SetModel();

	void SetMean(const std::string &mean_file, const std::string &mean_value);

	void SetLabels(const std::string &label_file);

	void WrapInputLayer(std::vector<GpuMat> *input_channels);

	void Preprocess(const Mat &img,
					std::vector<GpuMat> *input_channels);

  private:
	GPUAllocator *allocator_;
	std::shared_ptr<InferenceEngine> engine_;
	nvinfer1::IExecutionContext *context_;
	GpuMat mean_;
	std::vector<std::string> labels_;
	nvinfer1::DimsCHW input_dim_;
	Size input_cv_size_;
	float *input_layer_;
	nvinfer1::DimsCHW output_dim_;
	float *output_layer_;
};

#endif