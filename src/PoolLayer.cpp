#include "PoolLayer.hpp"
#include <limits>
#include "ConvOps.hpp"
namespace MaskedCNN {

PoolLayer::PoolLayer(int windowSize, std::string name)
    : windowSize(windowSize)
{
    this->name = name;
}

void PoolLayer::forwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();
    const Tensor<float> &prevMask = *bottoms[0]->getMask();


    if (!initDone)
    {
        auto dims = input.dimensions();

        channels = dims[0];
        inputHeight = dims[1];
        inputWidth = dims[2];
        outputHeight = std::floor((inputHeight - windowSize) / (double)windowSize + 1);
        outputWidth = std::floor((inputWidth - windowSize) / (double)windowSize + 1);
        output.resize({channels, outputHeight, outputWidth});
        delta.resize({channels, outputHeight, outputWidth});
        mask.resize({outputHeight, outputWidth});

        initDone = true;
    }

    mask.zero();
    convolveMaskIm2Col(prevMask, mask, buf, windowSize, windowSize, 0);

    for (int j = 0; j < outputHeight; j++)
    {
        for (int k = 0; k < outputWidth; k++)
        {
            if (maskEnabled && mask(j,k) == 0) continue;
            for (int i = 0; i < output.channelLength(); i++)
            {
                float max_float = std::numeric_limits<float>::lowest();
                for (int dy = 0; dy < windowSize; dy++)
                {
                    for (int dx = 0; dx < windowSize; dx++)
                    {
                        int y = j * windowSize + dy;
                        int x = k * windowSize + dx;

                        float currentEl = 0;

                        if (y >= 0 && y < inputHeight && x >= 0 && x < inputWidth)
                        {
                            currentEl = input(i, j * windowSize + dy, k * windowSize + dx);
                        }

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

void PoolLayer::backwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();
    Tensor<float> &prevDelta = *bottoms[0]->getDelta();

    assert(prevDelta.dimensions() == std::vector<int>({channels, inputHeight, inputWidth}));

    for (int i = 0; i < output.channelLength(); i++)
    {
        for (int j = 0; j < output.columnLength(); j++)
        {
            for (int k = 0; k < output.rowLength(); k++)
            {
                float maxEl = output(i, j, k);
                for (int dy = 0; dy < windowSize; dy++)
                {
                    for (int dx = 0; dx < windowSize; dx++)
                    {
                        if (maxEl == input(i, j * windowSize + dy, k * windowSize + dx))
                        {
                            prevDelta(i, j * windowSize + dy, k * windowSize + dx) =
                                    delta(i, j, k);
                        }
                        else
                        {
                            prevDelta(i, j * windowSize + dy, k * windowSize + dx) = 0;
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


