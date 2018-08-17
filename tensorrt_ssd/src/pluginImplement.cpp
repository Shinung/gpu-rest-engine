#include <vector>
#include <algorithm>
#include <cstring>

#include <glog/logging.h>

#include "pluginImplement.h"

/******************************/
// PluginFactory
/******************************/
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    CHECK(isPlugin(layerName));

    if (!strcmp(layerName, "conv4_3_norm"))
    {
        CHECK(mNormalizeLayer.get() == nullptr);
        mNormalizeLayer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDNormalizePlugin(weights, false, false, 0.001), nvPluginDeleter);//eps设置为0.0001
        return mNormalizeLayer.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_conf_perm"))
    {
        CHECK(mConv4_3_norm_mbox_conf_perm_layer.get() == nullptr);
        mConv4_3_norm_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv4_3_norm_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_loc_perm"))
    {
        CHECK(mConv4_3_norm_mbox_loc_perm_layer.get() == nullptr);
        mConv4_3_norm_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv4_3_norm_mbox_loc_perm_layer.get();
    }
    //ssd_pruning
    else if (!strcmp(layerName, "fc7_mbox_conf_perm"))
    {
        CHECK(mFc7_mbox_conf_perm_layer.get() == nullptr);
        mFc7_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mFc7_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_loc_perm"))
    {
        CHECK(mFc7_mbox_loc_perm_layer.get() == nullptr);
        mFc7_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mFc7_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_conf_perm"))
    {
        CHECK(mConv6_2_mbox_conf_perm_layer.get() == nullptr);
        mConv6_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv6_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_loc_perm"))
    {
        CHECK(mConv6_2_mbox_loc_perm_layer.get() == nullptr);
        mConv6_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv6_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_conf_perm"))
    {
        CHECK(mConv7_2_mbox_conf_perm_layer.get() == nullptr);
        mConv7_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv7_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_loc_perm"))
    {
        CHECK(mConv7_2_mbox_loc_perm_layer.get() == nullptr);
        mConv7_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv7_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_conf_perm"))
    {
        CHECK(mConv8_2_mbox_conf_perm_layer.get() == nullptr);
        mConv8_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv8_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_loc_perm"))
    {
        CHECK(mConv8_2_mbox_loc_perm_layer.get() == nullptr);
        mConv8_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv8_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_conf_perm"))
    {
        CHECK(mConv9_2_mbox_conf_perm_layer.get() == nullptr);
        mConv9_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv9_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_loc_perm"))
    {
        CHECK(mConv9_2_mbox_loc_perm_layer.get() == nullptr);
        mConv9_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv9_2_mbox_loc_perm_layer.get();
    }

    else if (!strcmp(layerName, "conv4_3_norm_mbox_priorbox"))
    {
        CHECK(mConv4_3_norm_mbox_priorbox_layer.get() == nullptr);
        float min_size = 30.0, max_size = 60.0, aspect_ratio[2] = {1.0, 2.0};
        mConv4_3_norm_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin({&min_size, &max_size, aspect_ratio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 8.0, 8.0, 0.5}), nvPluginDeleter);
        return mConv4_3_norm_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_priorbox"))
    {
        CHECK(mFc7_mbox_priorbox_layer.get() == nullptr);
        float min_size = 60.0, max_size = 111.0, aspect_ratio[3] = {1.0, 2.0, 3.0};
        mFc7_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin({&min_size, &max_size, aspect_ratio, 1, 1, 3, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 16.0, 16.0, 0.5}), nvPluginDeleter);
        return mFc7_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_priorbox"))
    {
        CHECK(mConv6_2_mbox_priorbox_layer.get() == nullptr);
        float min_size = 111.0, max_size = 162.0, aspect_ratio[3] = {1.0, 2.0, 3.0};
        mConv6_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin({&min_size, &max_size, aspect_ratio, 1, 1, 3, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 32.0, 32.0, 0.5}), nvPluginDeleter);
        return mConv6_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_priorbox"))
    {
        CHECK(mConv7_2_mbox_priorbox_layer.get() == nullptr);
        float min_size = 162.0, max_size = 213.0, aspect_ratio[3] = {1.0, 2.0, 3.0};
        mConv7_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin({&min_size, &max_size, aspect_ratio, 1, 1, 3, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 64.0, 64.0, 0.5}), nvPluginDeleter);
        return mConv7_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_priorbox"))
    {
        CHECK(mConv8_2_mbox_priorbox_layer.get() == nullptr);
        float min_size = 213.0, max_size = 264.0, aspect_ratio[2] = {1.0, 2.0};
        mConv8_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin({&min_size, &max_size, aspect_ratio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 100.0, 100.0, 0.5}), nvPluginDeleter);
        return mConv8_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_priorbox"))
    {
        CHECK(mConv9_2_mbox_priorbox_layer.get() == nullptr);
        float min_size = 264.0, max_size = 315.0, aspect_ratio[2] = {1.0, 2.0};
        mConv9_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin({&min_size, &max_size, aspect_ratio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 300.0, 300.0, 0.5}), nvPluginDeleter);
        return mConv9_2_mbox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "mbox_loc"))
    {
        CHECK(mBox_loc_layer.get() == nullptr);
        mBox_loc_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mBox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf"))
    {
        CHECK(mBox_conf_layer.get() == nullptr);
        mBox_conf_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return mBox_conf_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {
        CHECK(mBox_priorbox_layer.get() == nullptr);
        mBox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(2, true), nvPluginDeleter);
        return mBox_priorbox_layer.get();
    }
        //flatten
    else if (!strcmp(layerName, "conv4_3_norm_mbox_conf_flat"))
    {
        CHECK(mConv4_3_norm_mbox_conf_flat_layer.get() == nullptr);
        mConv4_3_norm_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv4_3_norm_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_loc_flat"))
    {
        CHECK(mConv4_3_norm_mbox_loc_flat_layer.get() == nullptr);
        mConv4_3_norm_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv4_3_norm_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_conf_flat"))
    {
        CHECK(mFc7_mbox_conf_flat_layer.get() == nullptr);
        mFc7_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mFc7_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_loc_flat"))
    {
        CHECK(mFc7_mbox_loc_flat_layer.get() == nullptr);
        mFc7_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mFc7_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_conf_flat"))
    {
        CHECK(mConv6_2_mbox_conf_flat_layer.get() == nullptr);
        mConv6_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv6_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_loc_flat"))
    {
        CHECK(mConv6_2_mbox_loc_flat_layer.get() == nullptr);
        mConv6_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv6_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_conf_flat"))
    {
        CHECK(mConv7_2_mbox_conf_flat_layer.get() == nullptr);
        mConv7_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv7_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_loc_flat"))
    {
        CHECK(mConv7_2_mbox_loc_flat_layer.get() == nullptr);
        mConv7_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv7_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_conf_flat"))
    {
        CHECK(mConv8_2_mbox_conf_flat_layer.get() == nullptr);
        mConv8_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv8_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_loc_flat"))
    {
        CHECK(mConv8_2_mbox_loc_flat_layer.get() == nullptr);
        mConv8_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv8_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_conf_flat"))
    {
        CHECK(mConv9_2_mbox_conf_flat_layer.get() == nullptr);
        mConv9_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv9_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_loc_flat"))
    {
        CHECK(mConv9_2_mbox_loc_flat_layer.get() == nullptr);
        mConv9_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mConv9_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf_flatten"))
    {
        CHECK(mMbox_conf_flat_layer.get() == nullptr);
        mMbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin());
        return mMbox_conf_flat_layer.get();
    }

        //reshape
    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {
        CHECK(mMbox_conf_reshape.get() == nullptr);
        CHECK(nbWeights == 0);
        CHECK(weights == nullptr);
        
        mMbox_conf_reshape = std::unique_ptr<ReshapePlugin>(new ReshapePlugin(mOutC));
        return mMbox_conf_reshape.get();
    }
    //softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax"))
    {
        CHECK(mPluginSoftmax == nullptr);
        CHECK(nbWeights == 0);
        CHECK(weights == nullptr);
        
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin());
        return mPluginSoftmax.get();
    }
    else if (!strcmp(layerName, "detection_out"))
    {
        CHECK(mDetection_out.get() == nullptr);
        mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
            (createSSDDetectionOutputPlugin({true, false, 0, mOutC, 400, 200, 0.5, 0.43, CodeType_t::CENTER_SIZE}), nvPluginDeleter);
        return mDetection_out.get();
    }
    else
    {
        LOG(FATAL) << layerName << std::endl;
    }

    return nullptr;
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    CHECK(isPlugin(layerName));

    if (!strcmp(layerName, "conv4_3_norm"))
    {
        CHECK(mNormalizeLayer.get() == nullptr);
        mNormalizeLayer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDNormalizePlugin(serialData, serialLength), nvPluginDeleter);
        return mNormalizeLayer.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_conf_perm"))
    {
        CHECK(mConv4_3_norm_mbox_conf_perm_layer.get() == nullptr);
        mConv4_3_norm_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv4_3_norm_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_loc_perm"))
    {
        CHECK(mConv4_3_norm_mbox_loc_perm_layer.get() == nullptr);
        mConv4_3_norm_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv4_3_norm_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_conf_perm"))
    {
        CHECK(mFc7_mbox_conf_perm_layer.get() == nullptr);
        mFc7_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mFc7_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_loc_perm"))
    {
        CHECK(mFc7_mbox_loc_perm_layer.get() == nullptr);
        mFc7_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mFc7_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_conf_perm"))
    {
        CHECK(mConv6_2_mbox_conf_perm_layer.get() == nullptr);
        mConv6_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv6_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_loc_perm"))
    {
        CHECK(mConv6_2_mbox_loc_perm_layer.get() == nullptr);
        mConv6_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv6_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_conf_perm"))
    {
        CHECK(mConv7_2_mbox_conf_perm_layer.get() == nullptr);
        mConv7_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv7_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_loc_perm"))
    {
        CHECK(mConv7_2_mbox_loc_perm_layer.get() == nullptr);
        mConv7_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv7_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_conf_perm"))
    {
        CHECK(mConv8_2_mbox_conf_perm_layer.get() == nullptr);
        mConv8_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv8_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_loc_perm"))
    {
        CHECK(mConv8_2_mbox_loc_perm_layer.get() == nullptr);
        mConv8_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv8_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_conf_perm"))
    {
        CHECK(mConv9_2_mbox_conf_perm_layer.get() == nullptr);
        mConv9_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv9_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_loc_perm"))
    {
        CHECK(mConv9_2_mbox_loc_perm_layer.get() == nullptr);
        mConv9_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return mConv9_2_mbox_loc_perm_layer.get();
    }

    else if (!strcmp(layerName, "conv4_3_norm_mbox_priorbox"))
    {
        CHECK(mConv4_3_norm_mbox_priorbox_layer.get() == nullptr);
        mConv4_3_norm_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mConv4_3_norm_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_priorbox"))
    {
        CHECK(mFc7_mbox_priorbox_layer.get() == nullptr);
        mFc7_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mFc7_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_priorbox"))
    {
        CHECK(mConv6_2_mbox_priorbox_layer.get() == nullptr);
        mConv6_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mConv6_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_priorbox"))
    {
        CHECK(mConv7_2_mbox_priorbox_layer.get() == nullptr);
        mConv7_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mConv7_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_priorbox"))
    {
        CHECK(mConv8_2_mbox_priorbox_layer.get() == nullptr);
        mConv8_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mConv8_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_priorbox"))
    {
        CHECK(mConv9_2_mbox_priorbox_layer.get() == nullptr);
        mConv9_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return mConv9_2_mbox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "mbox_loc"))
    {
        CHECK(mBox_loc_layer.get() == nullptr);
        mBox_loc_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mBox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf"))
    {
        CHECK(mBox_conf_layer.get() == nullptr);
        mBox_conf_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mBox_conf_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {
        CHECK(mBox_priorbox_layer.get() == nullptr);
        mBox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createConcatPlugin(serialData, serialLength), nvPluginDeleter);
        return mBox_priorbox_layer.get();
    }
        //flatten
    else if (!strcmp(layerName, "conv4_3_norm_mbox_conf_flat"))
    {
        CHECK(mConv4_3_norm_mbox_conf_flat_layer.get() == nullptr);
        mConv4_3_norm_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv4_3_norm_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_loc_flat"))
    {
        CHECK(mConv4_3_norm_mbox_loc_flat_layer.get() == nullptr);
        mConv4_3_norm_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv4_3_norm_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_conf_flat"))
    {
        CHECK(mFc7_mbox_conf_flat_layer.get() == nullptr);
        mFc7_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mFc7_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_loc_flat"))
    {
        CHECK(mFc7_mbox_loc_flat_layer.get() == nullptr);
        mFc7_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mFc7_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_conf_flat"))
    {
        CHECK(mConv6_2_mbox_conf_flat_layer.get() == nullptr);
        mConv6_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv6_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_loc_flat"))
    {
        CHECK(mConv6_2_mbox_loc_flat_layer.get() == nullptr);
        mConv6_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv6_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_conf_flat"))
    {
        CHECK(mConv7_2_mbox_conf_flat_layer.get() == nullptr);
        mConv7_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv7_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv7_2_mbox_loc_flat"))
    {
        CHECK(mConv7_2_mbox_loc_flat_layer.get() == nullptr);
        mConv7_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv7_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_conf_flat"))
    {
        CHECK(mConv8_2_mbox_conf_flat_layer.get() == nullptr);
        mConv8_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv8_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv8_2_mbox_loc_flat"))
    {
        CHECK(mConv8_2_mbox_loc_flat_layer.get() == nullptr);
        mConv8_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv8_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_conf_flat"))
    {
        CHECK(mConv9_2_mbox_conf_flat_layer.get() == nullptr);
        mConv9_2_mbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv9_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv9_2_mbox_loc_flat"))
    {
        CHECK(mConv9_2_mbox_loc_flat_layer.get() == nullptr);
        mConv9_2_mbox_loc_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mConv9_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf_flatten"))
    {
        CHECK(mMbox_conf_flat_layer.get() == nullptr);
        mMbox_conf_flat_layer = std::unique_ptr<FlattenPlugin>(new FlattenPlugin(serialData, serialLength));
        return mMbox_conf_flat_layer.get();
    }
    //reshape
    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {
        CHECK(mMbox_conf_reshape.get() == nullptr);
        mMbox_conf_reshape = std::unique_ptr<ReshapePlugin>(new ReshapePlugin(serialData, serialLength, mOutC));
        return mMbox_conf_reshape.get();
    }
    //softmax
    else if (!strcmp(layerName, "mbox_conf_softmax"))
    {
        CHECK(mPluginSoftmax.get() == nullptr);
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin(serialData, serialLength));
        return mPluginSoftmax.get();
    }

    else if (!strcmp(layerName, "detection_out"))
    {
        CHECK(mDetection_out.get() == nullptr);
        mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
        return mDetection_out.get();
    }
    else
    {
        LOG(FATAL) << layerName << std::endl;
    }

    return nullptr;
}

bool PluginFactory::isPlugin(const char* name)
{
    return (!strcmp(name, "conv4_3_norm")
            || !strcmp(name, "conv4_3_norm_mbox_conf_perm")
            || !strcmp(name, "conv4_3_norm_mbox_conf_flat")
            || !strcmp(name, "conv4_3_norm_mbox_loc_perm")
            || !strcmp(name, "conv4_3_norm_mbox_loc_flat")
            || !strcmp(name, "fc7_mbox_conf_perm")
            || !strcmp(name, "fc7_mbox_conf_flat")
            || !strcmp(name, "fc7_mbox_loc_perm")
            || !strcmp(name, "fc7_mbox_loc_flat")
            || !strcmp(name, "conv6_2_mbox_conf_perm")
            || !strcmp(name, "conv6_2_mbox_conf_flat")
            || !strcmp(name, "conv6_2_mbox_loc_perm")
            || !strcmp(name, "conv6_2_mbox_loc_flat")
            || !strcmp(name, "conv7_2_mbox_conf_perm")
            || !strcmp(name, "conv7_2_mbox_conf_flat")
            || !strcmp(name, "conv7_2_mbox_loc_perm")
            || !strcmp(name, "conv7_2_mbox_loc_flat")
            || !strcmp(name, "conv8_2_mbox_conf_perm")
            || !strcmp(name, "conv8_2_mbox_conf_flat")
            || !strcmp(name, "conv8_2_mbox_loc_perm")
            || !strcmp(name, "conv8_2_mbox_loc_flat")
            || !strcmp(name, "conv9_2_mbox_conf_perm")
            || !strcmp(name, "conv9_2_mbox_conf_flat")
            || !strcmp(name, "conv9_2_mbox_loc_perm")
            || !strcmp(name, "conv9_2_mbox_loc_flat")
            || !strcmp(name, "conv4_3_norm_mbox_priorbox")
            || !strcmp(name, "fc7_mbox_priorbox")
            || !strcmp(name, "conv6_2_mbox_priorbox")
            || !strcmp(name, "conv7_2_mbox_priorbox")
            || !strcmp(name, "conv8_2_mbox_priorbox")
            || !strcmp(name, "conv9_2_mbox_priorbox")
            || !strcmp(name, "mbox_conf_reshape")
            || !strcmp(name, "mbox_conf_flatten")
            || !strcmp(name, "mbox_loc")
            || !strcmp(name, "mbox_conf")
            || !strcmp(name, "mbox_priorbox")
            || !strcmp(name, "mbox_conf_softmax")
            || !strcmp(name, "detection_out"));
}

void PluginFactory::destroyPlugin()
{
    mNormalizeLayer.release();
    mNormalizeLayer = nullptr;

    mConv4_3_norm_mbox_conf_perm_layer.release();
    mConv4_3_norm_mbox_conf_perm_layer = nullptr;
    mConv4_3_norm_mbox_loc_perm_layer.release();
    mConv4_3_norm_mbox_loc_perm_layer = nullptr;
    mFc7_mbox_conf_perm_layer.release();
    mFc7_mbox_conf_perm_layer = nullptr;
    mFc7_mbox_loc_perm_layer.release();
    mFc7_mbox_loc_perm_layer = nullptr;
    mConv6_2_mbox_conf_perm_layer.release();
    mConv6_2_mbox_conf_perm_layer = nullptr;
    mConv6_2_mbox_loc_perm_layer.release();
    mConv6_2_mbox_loc_perm_layer = nullptr;
    mConv7_2_mbox_conf_perm_layer.release();
    mConv7_2_mbox_conf_perm_layer = nullptr;
    mConv7_2_mbox_loc_perm_layer.release();
    mConv7_2_mbox_loc_perm_layer = nullptr;
    mConv8_2_mbox_conf_perm_layer.release();
    mConv8_2_mbox_conf_perm_layer = nullptr;
    mConv8_2_mbox_loc_perm_layer.release();
    mConv8_2_mbox_loc_perm_layer = nullptr;
    mConv9_2_mbox_conf_perm_layer.release();
    mConv9_2_mbox_conf_perm_layer = nullptr;
    mConv9_2_mbox_loc_perm_layer.release();
    mConv9_2_mbox_loc_perm_layer = nullptr;

    mConv4_3_norm_mbox_priorbox_layer.release();
    mConv4_3_norm_mbox_priorbox_layer = nullptr;
    mFc7_mbox_priorbox_layer.release();
    mFc7_mbox_priorbox_layer = nullptr;
    mConv6_2_mbox_priorbox_layer.release();
    mConv6_2_mbox_priorbox_layer = nullptr;
    mConv7_2_mbox_priorbox_layer.release();
    mConv7_2_mbox_priorbox_layer = nullptr;
    mConv8_2_mbox_priorbox_layer.release();
    mConv8_2_mbox_priorbox_layer = nullptr;
    mConv9_2_mbox_priorbox_layer.release();
    mConv9_2_mbox_priorbox_layer = nullptr;

    mBox_loc_layer.release();
    mBox_loc_layer = nullptr;
    mBox_conf_layer.release();
    mBox_conf_layer = nullptr;
    mBox_priorbox_layer.release();
    mBox_priorbox_layer = nullptr;

    mConv4_3_norm_mbox_conf_flat_layer.release();
    mConv4_3_norm_mbox_conf_flat_layer = nullptr;
    mConv4_3_norm_mbox_loc_flat_layer.release();
    mConv4_3_norm_mbox_loc_flat_layer = nullptr;
    mFc7_mbox_conf_flat_layer.release();
    mFc7_mbox_conf_flat_layer = nullptr;
    mFc7_mbox_loc_flat_layer.release();
    mFc7_mbox_loc_flat_layer = nullptr;
    mConv6_2_mbox_conf_flat_layer.release();
    mConv6_2_mbox_conf_flat_layer = nullptr;
    mConv6_2_mbox_loc_flat_layer.release();
    mConv6_2_mbox_loc_flat_layer = nullptr;
    mConv7_2_mbox_conf_flat_layer.release();
    mConv7_2_mbox_conf_flat_layer = nullptr;
    mConv7_2_mbox_loc_flat_layer.release();
    mConv7_2_mbox_loc_flat_layer = nullptr;
    mConv8_2_mbox_conf_flat_layer.release();
    mConv8_2_mbox_conf_flat_layer = nullptr;
    mConv8_2_mbox_loc_flat_layer.release();
    mConv8_2_mbox_loc_flat_layer = nullptr;
    mConv9_2_mbox_conf_flat_layer.release();
    mConv9_2_mbox_conf_flat_layer = nullptr;
    mConv9_2_mbox_loc_flat_layer.release();
    mConv9_2_mbox_loc_flat_layer = nullptr;
    mMbox_conf_flat_layer.release();
    mMbox_conf_flat_layer = nullptr;

    mMbox_conf_reshape.release();
    mMbox_conf_reshape = nullptr;
    mPluginSoftmax.release();
    mPluginSoftmax = nullptr;
    mDetection_out.release();
    mDetection_out = nullptr;

}
