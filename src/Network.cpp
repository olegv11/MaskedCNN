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
#include "NetworkLoader.hpp"
#include "Visuals.hpp"


#include <openblas/cblas.h>


namespace MaskedCNN
{

Network::Network(std::vector<std::unique_ptr<Layer>> layers, int threshold)
    :layers(std::move(layers)), threshold(threshold)
{
    displayMaskSwitch.resize(this->layers.size());
    for (uint32_t i = 0; i < this->layers.size(); i++)
    {
        this->layers[i]->setTrainingMode(false);
        displayMaskSwitch[i] = false;
    }
}

Network::Network(std::string modelPath, int threshold)
    :layers(loadCaffeNet(modelPath)), maskEnabled(false),
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

void Network::setThreshold(int threshold)
{
    this->threshold = threshold;
}

std::vector<std::pair<std::string, cv::Mat>> Network::forward(const cv::Mat& input)
{
    Tensor<float> image;
    Tensor<float> mask;

    if (!initDone || prevFrame.size() != input.size())
    {
        mask.resize({input.rows, input.cols});
        image.resize({3, input.rows, input.cols});
        input.copyTo(prevFrame);
        accumMatrix.resize({input.rows, input.cols});
        accumMatrix.zero();
        initDone = true;
    }

    input.copyTo(currentFrame);
    mask = diffFrames(currentFrame, prevFrame, accumMatrix, threshold);
    image = matToTensor(currentFrame);
    image.add(-104.00699, -116.66877, -122.67892);
    std::cout << "Mask filled:" << mask.howFilled() << std::endl;
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
            std::cout << layers[i]->getName() << ":" << layers[i]->getMask()->howFilled() << std::endl;
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
    result.emplace_back("Result", cropLike(visualizeOutput(maxarg(*layers.back()->getOutput())), currentFrame, 8));
    return result;
}

void Network::dummyForward(const Tensor<float> &input, const Tensor<float>& mask)
{
    dynamic_cast<InputLayer*>(layers[0].get())->setInput(input);
    dynamic_cast<InputLayer*>(layers[0].get())->setMask(mask);

    times(&beginTime);
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->forwardPropagate();
    }
    times(&endTime);
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

Tensor<float> Network::getOutput()
{
    return cropLike(maxarg(*layers.back()->getOutput()), currentFrame, 8);
}

long Network::forwardTime() const
{
    return endTime.tms_utime + endTime.tms_stime - beginTime.tms_utime - beginTime.tms_stime;
}

}
