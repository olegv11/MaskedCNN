#pragma once
#include "Util.hpp"
#include "Layer.hpp"
#include "Activation.hpp"

#include <memory>

namespace MaskedCNN
{

class FullyConnectedLayer : public Layer
{
public:
    FullyConnectedLayer(int input, int neurons, std::unique_ptr<Activation> activation);
    virtual void forwardPropagate(const Tensor<float> &input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float> &prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;
    virtual int getNeuronInputNumber() const override;

private:
    std::unique_ptr<Activation> activation;
    int neurons;
    int inputCount;
};

}
