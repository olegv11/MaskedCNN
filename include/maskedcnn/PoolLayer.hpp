#pragma once
#include "Util.hpp"
#include "Layer.hpp"
#include "Activation.hpp"

namespace MaskedCNN
{

class PoolLayer : public Layer
{
public:
    PoolLayer(int windowWidth, int windowHeight, std::string name = "");

private:
    int channels;
    int inputHeight;
    int inputWidth;
    int windowWidth;
    int windowHeight;
    int outputHeight;
    int outputWidth;

    // Layer interface
public:
    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;
};

}

