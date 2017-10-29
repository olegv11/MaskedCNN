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
    if (!initDone)
    {
        auto inputDiemnsions = input.dimensions();
        output.resize(inputDiemnsions);
        delta.resize(inputDiemnsions);
    }

    output = input;
}

void InputLayer::setMask(const Tensor<float> mask)
{
    this->mask = mask;
}

}
