#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

#include <string>

#include "NvInfer.h"

enum class eNumOutClasses
{
  CLASS21 = 21
};

class InferenceEngine
{
  public:
    InferenceEngine(const std::string &model_file,
                    const std::string &trained_file,
                    eNumOutClasses out);

    ~InferenceEngine();

    nvinfer1::ICudaEngine *Get() const
    {
        return engine_;
    }

  private:
    nvinfer1::ICudaEngine *engine_;
};

#endif