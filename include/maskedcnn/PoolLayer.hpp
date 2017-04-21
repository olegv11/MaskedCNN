#pragma once
#include "Util.hpp"
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN
{

class PoolLayer : public Layer
{
public:
    PoolLayer(std::vector<int> dims, int windowWidth, int windowHeight);

private:
    int channels;
    int inputHeight;
    int inputWidth;
    int windowWidth;
    int windowHeight;

    // Layer interface
public:
    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;
};

}

