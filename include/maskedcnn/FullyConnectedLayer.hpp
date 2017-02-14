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
    FullyConnectedLayer(Tensor<float> &&weights, Tensor<float> &&biases, std::unique_ptr<Activation> activation);
    virtual void forwardPropagate(Tensor<float> &input) override;
    virtual void calculateGradients(const Tensor<float>& input) override;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float> &prevDelta) override;
    virtual std::vector<int> getOutputDimensions() override;

private:
    std::unique_ptr<Activation> activation;
    int neurons;
};

}
