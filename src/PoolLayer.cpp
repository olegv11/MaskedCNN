#include "PoolLayer.hpp"
#include <limits>
namespace MaskedCNN {

PoolLayer::PoolLayer(int channels, int inputWidth, int inputHeight, int windowWidth, int windowHeight)
    :channels(channels), inputWidth(inputWidth), inputHeight(inputHeight),
      windowWidth(windowWidth), windowHeight(windowHeight)
{
    output.resize({channels, inputHeight / windowHeight,inputWidth / windowWidth});
    delta.resize({channels, inputHeight / windowHeight,inputWidth / windowWidth});
}

void PoolLayer::forwardPropagate(Tensor<float>& input)
{
    for (int i = 0; i < output.channelLength(); i++)
    {
        for (int j = 0; j < output.columnLength(); j++)
        {
            for (int k = 0; k < output.rowLength(); k++)
            {
                float max_float = std::numeric_limits<float>::lowest();

                for (int dy = 0; dy < windowHeight; dy++)
                {
                    for (int dx = 0; dx < windowWidth; dx++)
                    {
                        float currentEl = input(i, j * windowHeight + dy, k * windowWidth + dx);
                        if (currentEl > max_float)
                        {
                            max_float = currentEl;
                        }
                    }
                }

                output(i,j,k) = max_float;
            }
        }
    }
}

void PoolLayer::calculateGradients(const Tensor<float>& input)
{
    (void)input;
    // no weights and biases to speak of
}

void PoolLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta)
{
    assert(prevDelta.dimensions() == std::vector<int>({channels, inputHeight, inputWidth}));

    for (int i = 0; i < output.channelLength(); i++)
    {
        for (int j = 0; j < output.columnLength(); j++)
        {
            for (int k = 0; k < output.rowLength(); k++)
            {
                float maxEl = output(i, j, k);
                for (int dy = 0; dy < windowHeight; dy++)
                {
                    for (int dx = 0; dx < windowWidth; dx++)
                    {
                        if (maxEl == input(i, j * windowHeight + dy, k * windowWidth + dx))
                        {
                            prevDelta(i, j * windowHeight + dy, k * windowWidth + dx) =
                                    delta(i, j, k);
                        }
                        else
                        {
                            prevDelta(i, j * windowHeight + dy, k * windowWidth + dx) = 0;
                        }
                    }
                }
            }
        }
    }
}

std::vector<int> PoolLayer::getOutputDimensions()
{
    return output.dimensions();
}


}


