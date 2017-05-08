#include "Network.hpp"
#include <string>
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <iomanip>
#include <array>
#include <deque>

#include <fstream>
#include "LayerLoader.hpp"
#include "Visuals.hpp"


#include <openblas/cblas.h>


namespace MaskedCNN
{

Network::Network(std::string modelPath, int threshold)
    :layers(loadCaffeNet(modelPath, 320, 240, 3)), maskEnabled(false),
      initDone(false), maskInitDone(false), threshold(threshold)
{
    displayMaskSwitch.resize(layers.size());
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->setTrainingMode(false);
        displayMaskSwitch[i] = false;
    }
}

void Network::setDisplayMask(int i, bool display)
{
    displayMaskSwitch[i] = display;
}

void Network::setDisplayMask(std::string name, bool display)
{
    auto l = std::find_if(layers.begin(), layers.end(),
                          [&](std::unique_ptr<Layer> const& layer){return layer->getName() == name;});
    if (l != layers.end())
    {
        displayMaskSwitch[l - layers.begin()] = display;
    }
}

void Network::setMaskEnabled(bool enabled)
{
    maskEnabled = enabled;
    maskInitDone = false;

    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->setMaskEnabled(false);
    }
}

std::vector<std::pair<std::string, cv::Mat>> Network::forward(const cv::Mat input)
{
    Tensor<float> image;
    Tensor<float> mask;

    if (!initDone)
    {
        mask.resize({input.rows, input.cols});
        image.resize({3, input.rows, input.cols});
        input.copyTo(prevFrame);
        initDone = true;
    }

    input.copyTo(currentFrame);
    mask = diffFrames(currentFrame, prevFrame, threshold);
    image = matToTensor(currentFrame);
    image.add(-104.00699, -116.66877, -122.67892);

    dynamic_cast<InputLayer*>(layers[0].get())->setInput(image);
    dynamic_cast<InputLayer*>(layers[0].get())->setMask(mask);

    times(&beginTime);
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->forwardPropagate();
    }
    times(&endTime);

    std::vector<std::pair<std::string, cv::Mat>> result;

    for (uint32_t i = 0; i < layers.size(); i++)
    {
        if (displayMaskSwitch[i])
        {
            result.emplace_back(layers[i]->displayMask());
        }
    }

    currentFrame.copyTo(prevFrame);

    if (maskEnabled && !maskInitDone)
    {
        for (uint32_t i = 0; i < layers.size(); i++)
        {
            layers[i]->setMaskEnabled(true);
        }
        maskInitDone = true;
    }
    result.emplace_back("Result", cropLike(visualizeOutput(maxarg(*layers.back()->getOutput())), currentFrame, 19));
    return result;
}

std::vector<std::string> Network::layerNames() const
{
    std::vector<std::string> result;

    for (const auto& l: layers)
    {
        std::string name = l->getName();
        if (name.find("conv") != std::string::npos
                || name.find("fc") != std::string::npos
                || name.find("score_fr") != std::string::npos
                || name.find("pool") != std::string::npos)
            result.emplace_back(name);
    }

    return result;
}

long Network::forwardTime() const
{
    return endTime.tms_utime + endTime.tms_stime - beginTime.tms_utime - beginTime.tms_stime;
}

}
