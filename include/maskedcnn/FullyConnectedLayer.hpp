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

private:
    Tensor<float> weights;
    Tensor<float> biases;
    std::unique_ptr<Activation> activation;
};

}
