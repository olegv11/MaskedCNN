#pragma once
#include "Layer.hpp"
#include "InputLayer.hpp"
#include "ConvolutionalLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "DropoutLayer.hpp"
#include "PoolLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "Activation.hpp"
#include "TrainingRegime.hpp"
#include "DataLoader.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>



#include <vector>
#include <sys/times.h>

namespace MaskedCNN
{

class Network
{
public:
    Network(std::vector<std::unique_ptr<Layer>> layers, int threshold);
    Network(std::string modelPath, int threshold);
    void setDisplayMask(int i, bool display);
    void setDisplayMask(std::string name, bool display);
    void setMaskEnabled(bool enabled);
    void setThreshold(int threshold);
    std::vector<std::pair<std::string, cv::Mat>> forward(const cv::Mat &input);
    void dummyForward(const Tensor<float> &input, const Tensor<float> &mask);
    std::vector<std::string> layerNames() const;
    Tensor<float> getOutput();

    long forwardTime() const;

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<bool> displayMaskSwitch;
    bool maskEnabled;

    cv::Mat currentFrame;
    cv::Mat prevFrame;
    bool initDone;
    bool maskInitDone;

    tms beginTime;
    tms endTime;

    int threshold;

    Tensor<float> accumMatrix;
};

}
