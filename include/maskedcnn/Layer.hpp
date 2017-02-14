#pragma once

#include "Tensor.hpp"
namespace MaskedCNN
{

class Layer
{
public:
    Layer();
    Layer(Tensor<float>&& weights, Tensor<float>&& biases);
    virtual ~Layer() = default;
    virtual void forwardPropagate(Tensor<float>& input) = 0;
    virtual void calculateGradients(const Tensor<float>& input) = 0;
    virtual void backwardPropagate(Tensor<float>& prevDelta) = 0;
    virtual std::vector<int> getOutputDimensions() = 0;

protected:
    Tensor<float> z;
    Tensor<float> weights;
    Tensor<float> biases;
    Tensor<float> output;
    Tensor<float> weight_delta; // dE/dw
    Tensor<float> bias_delta; // dE/db
    Tensor<float> delta; //dE/dz
    Tensor<float> dy_dz;
};

}
