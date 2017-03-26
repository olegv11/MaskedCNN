#pragma once
#include "Layer.hpp"

namespace MaskedCNN {

class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer(int numClasses);

    virtual void forwardPropagate(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float>& prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;

    float getLoss() const;
    void setGroundTruth(int i);

    int groundTruth;
    int numClasses;

    float loss;
};

}
