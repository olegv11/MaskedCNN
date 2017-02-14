#pragma once
#include "Util.hpp"
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN
{

class PoolLayer : public Layer
{
public:
    PoolLayer(int channels, int inputWidth, int inputHeight, int windowWidth, int windowHeight);

private:
    int channels;
    int inputWidth;
    int inputHeight;
    int windowWidth;
    int windowHeight;

    // Layer interface
public:
    virtual void forwardPropagate(Tensor<float>& input) override;
    virtual void calculateGradients(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;
};

}

