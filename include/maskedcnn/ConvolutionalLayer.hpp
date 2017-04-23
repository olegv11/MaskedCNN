#pragma once
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN {


class BaseConvolutionalLayer : public Layer
{
public:
    BaseConvolutionalLayer(std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int filterDepth, int featureMaps);
    virtual std::vector<int> getOutputDimensions() override;
    virtual int getNeuronInputNumber() const override;

protected:
    std::unique_ptr<Activation> activation;
    int pad;
    int stride;
    int filterSize;
    int filterDepth;
    int outputWidth, outputHeight, outputChannels;
    int inputWidth, inputHeight;
};

class ConvolutionalLayer : public BaseConvolutionalLayer
{
public:
    ConvolutionalLayer(std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int filterDepth, int featureMaps);
    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float>& prevDelta) override;
};

class DeconvolutionalLayer : public BaseConvolutionalLayer
{
public:
    DeconvolutionalLayer(std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int filterDepth, int featureMaps);
    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float>& prevDelta) override;
};

}
