#include "FullyConnectedLayer.hpp"


namespace MaskedCNN {

// Weight: [Neuron Count x Input count]
// Bias: [Neuron Count]

FullyConnectedLayer::FullyConnectedLayer(Tensor<float>&& weights, Tensor<float>&& biases, std::unique_ptr<Activation> activation)
    :Layer(std::move(weights), std::move(biases)), activation(std::move(activation))
{
    neurons = weights.columnLength();
    assert(biases.dimensionCount() == 1);
    assert(biases.elementCount() == neurons);

    z.resize({ 1, 1, neurons });
    dy_dz.resize({ 1, 1, neurons });
    delta.resize({ 1, 1, neurons });
    output.resize({ 1, 1, neurons });

    weight_delta.resize(weights.dimensions());
    bias_delta.resize(biases.dimensions());
}


void FullyConnectedLayer::forwardPropagate(Tensor<float> &input)
{
    Tensor<float> flatInput(input, shallow_copy{});
    flatInput.flatten();
    assert(flatInput.elementCount() == weights.columnLength());

    cblas_sgemv(CblasRowMajor, CblasNoTrans, weights.rowLength(), weights.columnLength(), 1.0, // z = w*prev_input + b
                &weights[0], weights.columnLength(), &flatInput[0], 1, 0.0, &z[0], 1);

    activation->activate(&z[0], &output[0], &dy_dz[0], neurons);
    for (int neuron = 0; neuron < neurons; neuron++)
    {
        z[neuron] += biases[neuron];        
    }
}

// delta should be equal to de/dy by now
void FullyConnectedLayer::calculateGradients(const Tensor<float> &input)
{
    Tensor<float> flatInput(input, shallow_copy{});
    flatInput.flatten();
    assert(flatInput.elementCount() == weights.columnLength());

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

}

void FullyConnectedLayer::backwardPropagate(const Tensor<float> &input, Tensor<float> &prevDelta)
{
    (void)(input);
    Tensor<float> flatDelta(prevDelta, shallow_copy{});
    flatDelta.flatten();

    assert(flatDelta.elementCount() == weights.columnLength());

    flatDelta.fillwith(0.0);

    cblas_sgemv(CblasRowMajor, CblasTrans, weights.rowLength(), weights.columnLength(), 1.0,
                &weights[0], weights.columnLength(), &delta[0], 1, 0.0, &flatDelta[0], 1); // setting previous de/dy

}

std::vector<int> FullyConnectedLayer::getOutputDimensions()
{
    return { 1, 1, neurons };
}


}

