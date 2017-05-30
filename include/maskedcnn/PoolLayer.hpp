#pragma once
#include "Util.hpp"
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN
{

class PoolLayer : public Layer
{
public:
    PoolLayer(int windowSize, std::string name = "");
    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;

private:
    int channels;
    int inputHeight;
    int inputWidth;
    int windowSize;
    int outputHeight;
    int outputWidth;
    Tensor<float> buf;
};

}

