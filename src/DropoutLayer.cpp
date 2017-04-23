#include "DropoutLayer.hpp"
#include <random>

namespace MaskedCNN {

static std::random_device rd;
static std::mt19937 gen(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);


DropoutLayer::DropoutLayer(double dropProbability)
    : dropProbability(dropProbability)
{
}


void DropoutLayer::forwardPropagate(const Tensor<float>& input)
{
    auto dims = input.dimensions();
    output.resize(dims);
    dropped.resize({output.elementCount()});
    delta.resize(dims);


    Tensor<float> flatInput(input, shallow_copy{});
    Tensor<float> flatOutput(output, shallow_copy{});
    int elementCount = flatInput.elementCount();

    if (isTraining)
    {
        flatOutput.zero();

        for (int i = 0; i < elementCount; i++)
        {
            if (distr(gen) > dropProbability)
            {
                dropped[i] = false;
                flatOutput[i] = flatInput[i];
            }
            else
            {
                dropped[i] = true;
            }
        }
    }
    else
    {
        for (int i = 0; i < elementCount; i++)
        {
            flatOutput[i] = flatInput[i] * (1 - dropProbability);
        }
    }

}

void DropoutLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta)
{
    (void)input;
    Tensor<float> flatPrevDelta(prevDelta, shallow_copy{});
    Tensor<float> flatDelta(delta, shallow_copy{});
    int elementCount = flatDelta.elementCount();

    flatPrevDelta.zero();

    for (int i = 0; i < elementCount; i++)
    {
        if (dropped[i] == false)
        {
            flatPrevDelta[i] = flatDelta[i];
        }
    }
}

std::vector<int> DropoutLayer::getOutputDimensions()
{
    return output.dimensions();
}


}



