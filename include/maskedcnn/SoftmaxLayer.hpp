#pragma once
#include "Layer.hpp"

namespace MaskedCNN {

class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer(int numClasses);

    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;

    float getLoss() const;
    void setGroundTruth(int i);

    int groundTruth;
    int numClasses;

    float loss;
};

}
