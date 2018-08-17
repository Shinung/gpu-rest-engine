#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <memory>

#include "NvCaffeParser.h"

#include "plugin/ReshapePlugin.h"
#include "plugin/FlattenPlugin.h"
#include "plugin/SoftmaxPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvinfer1::plugin;

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
    explicit PluginFactory(int outC) : mOutC(outC) { }
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

    void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };

    bool isPlugin(const char* name) override;
    void destroyPlugin();
    //normalize layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mNormalizeLayer{ nullptr, nvPluginDeleter };
    //permute layers
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv4_3_norm_mbox_conf_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv4_3_norm_mbox_loc_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mFc7_mbox_conf_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mFc7_mbox_loc_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv6_2_mbox_conf_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv6_2_mbox_loc_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv7_2_mbox_conf_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv7_2_mbox_loc_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv8_2_mbox_conf_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv8_2_mbox_loc_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv9_2_mbox_conf_perm_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv9_2_mbox_loc_perm_layer{ nullptr, nvPluginDeleter };
    //priorbox layers
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv4_3_norm_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mFc7_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv6_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv7_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv8_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv9_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
    //detection output layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mDetection_out{ nullptr, nvPluginDeleter };
    //concat layers
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mBox_loc_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mBox_conf_layer{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mBox_priorbox_layer{ nullptr, nvPluginDeleter };
    //reshape layer
    std::unique_ptr<ReshapePlugin> mMbox_conf_reshape{ nullptr };
    //flatten layers
    std::unique_ptr<FlattenPlugin> mConv4_3_norm_mbox_conf_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv4_3_norm_mbox_loc_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mFc7_mbox_conf_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mFc7_mbox_loc_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv6_2_mbox_conf_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv6_2_mbox_loc_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv7_2_mbox_conf_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv7_2_mbox_loc_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv8_2_mbox_conf_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv8_2_mbox_loc_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv9_2_mbox_conf_flat_layer{ nullptr };
    std::unique_ptr<FlattenPlugin> mConv9_2_mbox_loc_flat_layer{ nullptr };
    //softmax layer
    std::unique_ptr<SoftmaxPlugin> mPluginSoftmax{ nullptr };
    std::unique_ptr<FlattenPlugin> mMbox_conf_flat_layer{ nullptr };

private:
    int mOutC;
};

#endif
