 #include "InputLayer.hpp"

namespace MaskedCNN
{

void InputLayer::forwardPropagate()
{
}

// Doesn't make sense to backprop from the first layer
void InputLayer::backwardPropagate()
{
    assert(false);
}

std::vector<int> InputLayer::getOutputDimensions()
{
    return output.dimensions();
}

void InputLayer::setInput(const Tensor<float> input)
{
    output = input;
}

}
