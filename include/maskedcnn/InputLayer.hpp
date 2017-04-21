#pragma once
#include "Layer.hpp"

namespace MaskedCNN
{

class InputLayer : public Layer {

public:
    InputLayer(const std::vector<int> inputDimensions)
    : Layer()
    {
        output.resize(inputDimensions);
        delta.resize(inputDimensions);
    }

    // Layer interface
public:
    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;
};

}
