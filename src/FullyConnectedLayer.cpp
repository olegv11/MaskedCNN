#include "FullyConnectedLayer.hpp"


namespace MaskedCNN {

FullyConnectedLayer::FullyConnectedLayer(Tensor<float>&& weights, Tensor<float>&& biases, std::unique_ptr<Activation> activation)
    : weights(std::move(weights)), biases(std::move(biases)),
      activation(std::move(activation))
{
}


}
