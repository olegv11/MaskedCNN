#include "Layer.hpp"

namespace MaskedCNN
{

Layer::Layer()
{

}

Layer::Layer(Tensor<float> &&weights, Tensor<float> &&biases)
    :weights(std::move(weights)), biases(std::move(biases))
{
}

}
