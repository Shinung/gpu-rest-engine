#include <iostream>
#include <fstream>

#include "detection.h"

#define MODEL "/home/qisens/roy/gpu-rest-engine/models/VGGNet/VOC0712/SSD_300x300/ssd_deploy_iplugin.prototxt"
#define TRAINED "/home/qisens/roy/gpu-rest-engine/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"
#define LABEL "/home/qisens/caffe_ssd_origin/caffe/data/VOC0712/labelmap_voc.prototxt"

int main(int argc, char** argv)
{
    detection_ctx* ctx;

    ctx = detection_initialize(MODEL, TRAINED, "", "104,117,123", LABEL);

    {
        char* buffer;
        int len;
        std::ifstream inFile;

        inFile.open("/home/qisens/caffe_ssd_origin/caffe/examples/images/fish-bike.jpg", std::ios::in | std::ios::binary);

        inFile.seekg(0, std::ios::end);
        len = inFile.tellg();
        inFile.seekg(0, std::ios::beg);

        buffer = new char[len];
        inFile.read(buffer, len);
        inFile.close();

        const char* res;
        res = detection_inference(ctx, buffer, len);

        std::cout << res << std::endl;

        delete[] buffer;
        free((void*)res);
    }

    detection_destroy(ctx);

    return 0;
}