#pragma once
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN {

class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer(std::vector<int> dims, std::unique_ptr<Activation> activation, int stride,
                       int filterSize, int pad, int featureMaps);

    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;
    virtual int getNeuronInputNumber() const override;

private:
    std::unique_ptr<Activation> activation;
    int pad;
    int stride;
    int filterSize;
    int filterDepth;
    int outputWidth, outputHeight, outputChannels;
    int inputWidth, inputHeight;

};

}
