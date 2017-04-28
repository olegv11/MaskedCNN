#include "PoolLayer.hpp"
#include <limits>
namespace MaskedCNN {

PoolLayer::PoolLayer(int windowWidth, int windowHeight, std::string name)
    : windowWidth(windowWidth), windowHeight(windowHeight)
{
    this->name = name;
}

void PoolLayer::forwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();
    const Tensor<float> &prevMask = *bottoms[0]->getMask();

    //std::cout << "Forward start " << name << std::endl;

    if (!initDone)
    {
        auto dims = input.dimensions();

        channels = dims[0];
        inputHeight = dims[1];
        inputWidth = dims[2];
        outputHeight = (inputHeight + windowHeight - 1) / windowHeight;
        outputWidth = (inputWidth + windowWidth - 1) / windowWidth;
        output.resize({channels, outputHeight, outputWidth});
        delta.resize({channels, outputHeight, outputWidth});
        mask.resize({outputHeight, outputWidth});

        initDone = true;
    }

    mask.zero();
    for (int i = 0; i < output.channelLength(); i++)
    {
        for (int j = 0; j < outputHeight; j++)
        {
            for (int k = 0; k < outputWidth; k++)
            {
                float max_float = std::numeric_limits<float>::lowest();
                bool changed = false;
                for (int dy = 0; dy < windowHeight; dy++)
                {
                    for (int dx = 0; dx < windowWidth; dx++)
                    {
                        int y = j * windowHeight + dy;
                        int x = k * windowWidth + dx;

                        float currentEl = 0;

                        if (y >= 0 && y < inputHeight && x >= 0 && x < inputWidth)
                        {
                            currentEl = input(i, j * windowHeight + dy, k * windowWidth + dx);

                            if (maskEnabled && prevMask(y,x) > 0)
                            {
                                changed = true;
                            }
                        }

                        if (currentEl > max_float)
                        {
                            max_float = currentEl;
                        }
                    }
                }
                if (maskEnabled)
                {
                    if (changed)
                    {
                        mask(j,k) = 1;
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


