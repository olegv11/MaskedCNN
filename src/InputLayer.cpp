 #include "InputLayer.hpp"

namespace MaskedCNN
{

void InputLayer::forwardPropagate(const Tensor<float>& input)
{
    assert(output.dimensions() == input.dimensions());
    output = input;
}

// Doesn't make sense to backprop from the first layer
void InputLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta)
{
    (void)input;
    (void) prevDelta;
    assert(false);
}

std::vector<int> InputLayer::getOutputDimensions()
{
    return output.dimensions();
}

}
