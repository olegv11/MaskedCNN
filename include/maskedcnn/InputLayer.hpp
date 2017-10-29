#pragma once
#include "Layer.hpp"

namespace MaskedCNN
{

class InputLayer : public Layer {

public:
    InputLayer(std::string name)
    : Layer()
    {
        this->name = name;
    }

    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;

    void setInput(const Tensor<float> input);
    void setMask(const Tensor<float> mask);
};

}
