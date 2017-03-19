#pragma once
#include <memory>
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN {

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer(int width, int height, int inputChannels, std::unique_ptr<Activation> activation, int pad, int stride,
                       int filterSize, int featureMaps);

    virtual void forwardPropagate(Tensor<float>& input) override;
    virtual void calculateGradients(const Tensor<float> &input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;

private:
    std::unique_ptr<Activation> activation;
    int pad;
    int stride;
    int filterSize;
    int outputSizeX, outputSizeY, outputChannels;

};

}
