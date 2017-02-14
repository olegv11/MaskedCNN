#pragma once

#include "Tensor.hpp"
namespace MaskedCNN
{

class Layer
{
public:
    Layer();
    virtual ~Layer() = default;
    virtual void forwardPropagate(Tensor<float>& input) = 0;
    //virtual void backwardPropagate(const Tensor<float>& input,  Tensor<float>& prevDelta);
    virtual std::vector<int> getOutputDimensions() = 0;

private:

};

}
