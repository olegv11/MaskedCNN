#pragma once
#include "Layer.hpp"


namespace MaskedCNN
{

// Currently only sum is implemented
class EltwiseLayer : public Layer
{
public:
    EltwiseLayer(std::string name = "");

    // Layer interface
public:
    virtual void forwardPropagate() override;
    virtual void backwardPropagate() override;
    virtual std::vector<int> getOutputDimensions() override;
};

void EltwiseLayer::forwardPropagate()
{
    output.resize(bottoms[0]->getOutput()->dimensions());
    delta.resize(bottoms[0]->getOutput()->dimensions());

    for (const auto& bottom : bottoms)
    {
        const auto& input = *bottom->getOutput();
        assert(input->dimensions() == output.dimensions());

        for (int c = 0; c < output.channelLength(); c++)
        {
            for (int y = 0; y < output.columnLength(); y++)
            {
                for (int x = 0; x < output.rowLength(); x++)
                {
                    output(c, y, x) += input(c, y, x);
                }
            }
        }
    }
}

void EltwiseLayer::backwardPropagate()
{
    assert(false);
}

std::vector<int> EltwiseLayer::getOutputDimensions()
{
    return output.dimensions();
}

}
