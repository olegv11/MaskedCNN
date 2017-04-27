#pragma once
#include "Layer.hpp"
#include "Activation.hpp"
#include "Tensor.hpp"

namespace MaskedCNN {


class BaseConvolutionalLayer : public Layer
{
public:
    BaseConvolutionalLayer(std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int filterDepth, int featureMaps, std::string name = "");
    BaseConvolutionalLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights,
                           Tensor<float>&& biases, int stride, int pad, std::string name = "");
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
    Tensor<float> colBuffer;
};

class ConvolutionalLayer : public BaseConvolutionalLayer
{
public:
    ConvolutionalLayer(std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int filterDepth, int featureMaps, std::string name = "");
    ConvolutionalLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights,
                           Tensor<float>&& biases, int stride, int pad, std::string name = "");
    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
};

class DeconvolutionalLayer : public BaseConvolutionalLayer
{
public:
    DeconvolutionalLayer(std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int filterDepth, int featureMaps, std::string name = "");
    DeconvolutionalLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights,
                           Tensor<float>&& biases, int stride, int pad, std::string name = "");
    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
};

}
