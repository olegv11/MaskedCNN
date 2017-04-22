#pragma once
#include "Layer.hpp"

namespace MaskedCNN {


// See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for details
class DropoutLayer : public Layer
{
public:
    DropoutLayer(std::vector<int> dims, double dropProbability);

    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;

private:
    double dropProbability;
    Tensor<bool> dropped;

};

}
