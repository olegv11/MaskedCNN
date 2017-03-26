#include "FullyConnectedLayer.hpp"

namespace MaskedCNN {

// Weight: [Neuron Count x Input count]
// Bias: [Neuron Count]

FullyConnectedLayer::FullyConnectedLayer(int inputCount, int neurons, std::unique_ptr<Activation> activation)
    :Layer(Tensor<float>({neurons, inputCount}), Tensor<float>({neurons})), activation(std::move(activation)), neurons(neurons)
{
    z.resize({ neurons });
    dy_dz.resize({ neurons });
    delta.resize({ neurons });
    output.resize({ neurons });

    weight_delta.resize(weights.dimensions());
    bias_delta.resize(biases.dimensions());
}


void FullyConnectedLayer::forwardPropagate(const Tensor<float> &input)
{
    Tensor<float> flatInput(const_cast<Tensor<float>&>(input), shallow_copy{});
    flatInput.flatten();
    assert(flatInput.elementCount() == weights.rowLength());

    cblas_sgemv(CblasRowMajor, CblasNoTrans, weights.columnLength(), weights.rowLength(), 1.0, // z = w*prev_input + b
                weights.dataAddress(), weights.rowLength(), flatInput.dataAddress(), 1, 0.0, z.dataAddress(), 1);

    for (int neuron = 0; neuron < neurons; neuron++)
    {
        z[neuron] += biases[neuron];
    }
    activation->activate(z.dataAddress(), output.dataAddress(), dy_dz.dataAddress(), neurons);
}


// delta should be equal to de/dy by now
void FullyConnectedLayer::backwardPropagate(const Tensor<float> &input, Tensor<float> &prevDelta)
{
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

    Tensor<float> flatDelta(prevDelta, shallow_copy{});
    flatDelta.flatten();

    assert(flatDelta.elementCount() == weights.rowLength());

    flatDelta.zero();

    cblas_sgemv(CblasRowMajor, CblasTrans, weights.columnLength(), weights.rowLength(), 1.0,
                weights.dataAddress(), weights.rowLength(), delta.dataAddress(), 1, 0.0, flatDelta.dataAddress(), 1); // setting previous de/dy

}

std::vector<int> FullyConnectedLayer::getOutputDimensions()
{
    return { neurons };
}


}

