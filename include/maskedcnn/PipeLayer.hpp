#pragma once
#include "Util.hpp"
#include "Layer.hpp"

namespace MaskedCNN
{

class PipeLayer : public Layer
{
public:
    PipeLayer(std::string name = "")
    {
        this->name = name;
    }

    virtual void forwardPropagate() override {}
    virtual void backwardPropagate() override {}
    virtual std::vector<int> getOutputDimensions() override
    {
        return bottoms[0]->getOutputDimensions();
    }

    virtual const Tensor<float> *getOutput() override
    {
        return bottoms[0]->getOutput();
    }

    virtual Tensor<float> *getDelta() override
    {
        return bottoms[0]->getDelta();
    }

    virtual Tensor<float> *getMask() override
    {
        return bottoms[0]->getMask();
    }
};


}
