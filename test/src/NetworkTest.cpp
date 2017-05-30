#include "gtest/gtest.h"
#include <Layer.hpp>
#include <Tensor.hpp>
#include <DataLoader.hpp>
#include <LayerLoader.hpp>
#include <Visuals.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

using namespace MaskedCNN;

namespace {

TEST(Network_Test, MaskedAndNormalNetworksAreEquivalent)
{
    std::vector<std::unique_ptr<Layer>> layersMasked;
    std::vector<std::unique_ptr<Layer>> layersNonMasked;

    layersMasked = loadCaffeNet("/home/oleg/Deep_learning/fcn/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel", 320, 240, 3);
    layersNonMasked = loadCaffeNet("/home/oleg/Deep_learning/fcn/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel", 320, 240, 3);

    for (uint32_t i = 0; i < layersNonMasked.size(); i++)
    {
        layersNonMasked[i]->setTrainingMode(false);
        layersMasked[i]->setTrainingMode(false);
    }

    cv::VideoCapture cap = cv::VideoCapture("/home/oleg/videoSmooth.avi");
    if (!cap.isOpened())
    {
        throw std::exception();
    }

    cv::Mat frame;
    cv::Mat prevFrame;
    cv::Mat accum;
    cap.read(prevFrame);

    for (int framesProcessed = 0; framesProcessed < 5; framesProcessed++)
    {
        Tensor<float> mask(std::vector<int>{prevFrame.rows, prevFrame.cols});
        Tensor<float> accum({prevFrame.rows, prevFrame.cols});
        cap.read(frame);
        mask = diffFrames(frame, prevFrame, accum);
        Tensor<float> image = matToTensor(frame);
        image.add(-104.00699, -116.66877, -122.67892);

        dynamic_cast<InputLayer*>(layersNonMasked[0].get())->setInput(image);
        dynamic_cast<InputLayer*>(layersMasked[0].get())->setInput(image);
        dynamic_cast<InputLayer*>(layersMasked[0].get())->setMask(mask);

        for (uint32_t i = 0; i < layersMasked.size(); i++)
        {
            auto& lNoMask = *layersNonMasked[i];
            lNoMask.forwardPropagate();

            auto& lMask = *layersMasked[i];
            lMask.forwardPropagate();

            if (i < 4)
            {
                lMask.setMaskEnabled(true);
            }
        }

        for (uint32_t i = 0; i < layersMasked.size(); i++)
        {
            ASSERT_EQ(*(layersNonMasked[i]->getOutput()), *(layersMasked[i]->getOutput())) << "Not equal at layer " << layersMasked[i]->getName();
        }
        std::cout << "Processed frame " << framesProcessed << std::endl;
    }
}


}
