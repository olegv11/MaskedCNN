#include "FullyConnectedLayer.hpp"

namespace MaskedCNN {

// Weight: [Neuron Count x Input count]
// Bias: [Neuron Count]

FullyConnectedLayer::FullyConnectedLayer(std::unique_ptr<Activation> activation, int neurons, std::string name)
    :activation(std::move(activation)), neurons(neurons)
{
    z.resize({ neurons });
    dy_dz.resize({ neurons });
    delta.resize({ neurons });
    output.resize({ neurons });
    this->name = name;
}

FullyConnectedLayer::FullyConnectedLayer(std::unique_ptr<Activation> activation, Tensor<float> &&weights, Tensor<float> &&biases, std::string name)
    :Layer(std::move(weights), std::move(biases), name) , activation(std::move(activation))
{
    neurons = this->weights.columnLength();
    inputCount = this->weights.rowLength();
    z.resize({ neurons });
    dy_dz.resize({ neurons });
    delta.resize({ neurons });
    output.resize({ neurons });
}


void FullyConnectedLayer::forwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();

    if (isTraining)
    {
        if (!initDone)
        {
            inputCount = multiplyAllElements(input.dimensions());

            weights.resize({neurons, inputCount});
            weight_delta.resize({neurons, inputCount});

            biases.resize({inputCount});
            bias_delta.resize({inputCount});

            initializeWeightsNormalDistrCorrectedVar();
            initDone = true;
        }
    }
    else
    {
        assert(inputCount == multiplyAllElements(input.dimensions()));
    }

    Tensor<float> flatInput(const_cast<Tensor<float>&>(input), shallow_copy{});
    flatInput.flatten();
    assert(flatInput.elementCount() == weights.rowLength());

    cblas_sgemv(CblasRowMajor, CblasNoTrans, weights.columnLength(), weights.rowLength(), 1.0, // z = w*flat_input + b
                weights.dataAddress(), weights.rowLength(), flatInput.dataAddress(), 1, 0.0, z.dataAddress(), 1);

    for (int neuron = 0; neuron < neurons; neuron++)
    {
        z[neuron] += biases[neuron];
    }
    activation->activate(z.dataAddress(), output.dataAddress(), dy_dz.dataAddress(), neurons);
}


// delta should be equal to de/dy by now
void FullyConnectedLayer::backwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();
    Tensor<float> &prevDelta = *bottoms[0]->getDelta();

    Tensor<float> flatInput(const_cast<Tensor<float>&>(input), shallow_copy{});
    flatInput.flatten();
    assert(flatInput.elementCount() == weights.rowLength());

    // de/dz = de/dy * dy/dz
    elementwiseMultiplication(&delta[0], &dy_dz[0], &delta[0], neurons);

    vectorCopy(&bias_delta[0], &delta[0], neurons);

    int inputCount = weights.rowLength();
    for (int i = 0; i < neurons; i++)
    {
        for (int j = 0; j < inputCount; j++)
        {
            weight_delta(i,j) = flatInput[j] * delta[i];
        }
    }

    assert(prevDelta.elementCount() == weights.rowLength());

    cblas_sgemv(CblasRowMajor, CblasTrans, weights.columnLength(), weights.rowLength(), 1.0,
                weights.dataAddress(), weights.rowLength(), delta.dataAddress(), 1, 0.0, prevDelta.dataAddress(), 1); // setting previous de/dy

}

std::vector<int> FullyConnectedLayer::getOutputDimensions()
{
    return { 1, 1, neurons };
}

int FullyConnectedLayer::getNeuronInputNumber() const
{
    return inputCount;
}


}

