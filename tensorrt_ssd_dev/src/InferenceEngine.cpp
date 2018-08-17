#include <glog/logging.h>

#include "InferenceEngine.h"
#include "pluginImplement.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;

class Logger : public nvinfer1::ILogger			
{
    public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) override
	{
		// suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: LOG(ERROR) << "INTERNAL_ERROR: " << msg; break;
            case Severity::kERROR: LOG(ERROR) << "ERROR: " << msg; break;
            case Severity::kWARNING: LOG(WARNING) << "WARNING: " << msg; break;
            case Severity::kINFO: LOG(INFO) << "INFO: " << msg; break;
            default: LOG(INFO) << "UNKNOWN: " << msg; break;
        }
	}
};
static Logger gLogger;

InferenceEngine::InferenceEngine(const string &model_file,
                                 const string &trained_file,
                                 eNumOutClasses out)
{
    PluginFactory pluginFactory(static_cast<int>(out));
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    const IBlobNameToTensor* blobNameToTensor = parser->parse(model_file.c_str(),
                                                              trained_file.c_str(),
                                                              *network, DataType::kFLOAT);

    CHECK(blobNameToTensor != nullptr);
    // specify which tensors are outputs
    //network->markOutput(*blobNameToTensor->find("detection_out"));
    network->markOutput(*blobNameToTensor->find("detection_out"));

    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(32 << 20);

    engine_ = builder->buildCudaEngine(*network);
    CHECK(engine_ != nullptr);

    network->destroy();
    parser->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
}

InferenceEngine::~InferenceEngine()
{
    engine_->destroy();
}