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
    FullyConnectedLayer(std::unique_ptr<Activation> activation, int neurons, std::string name = "");
    FullyConnectedLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights,
                           Tensor<float>&& biases, std::string name = "");
    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;
    virtual int getNeuronInputNumber() const override;

private:
    std::unique_ptr<Activation> activation;
    int neurons;
    int inputCount;
};

}
