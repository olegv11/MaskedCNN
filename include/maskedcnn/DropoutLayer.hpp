#pragma once
#include "Layer.hpp"

namespace MaskedCNN {


// See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for details
class DropoutLayer : public Layer
{
public:
    DropoutLayer(double dropProbability, std::string name = "");

    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;

private:
    double dropProbability;
    Tensor<bool> dropped;

};

}
