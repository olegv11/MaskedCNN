#pragma once

#include <memory>
#include "Tensor.hpp"
#include "TrainingRegime.hpp"
namespace MaskedCNN
{

class Layer
{
public:
    Layer();
    Layer(Tensor<float>&& weights, Tensor<float>&& biases);
    virtual ~Layer() = default;
    virtual void forwardPropagate(const Tensor<float>& input) = 0;
    virtual void backwardPropagate(const Tensor<float> &input, Tensor<float>& prevDelta) = 0;
    virtual std::vector<int> getOutputDimensions() = 0;
    virtual int getNeuronInputNumber() const { return 0; }

    const Tensor<float> *getOutput();
    Tensor<float> *getDelta();

    void initializeWeightsStandardDistr();
    void initializeWeightsNormalDistrCorrectedVar();

    void setTrainer(std::unique_ptr<TrainingRegime> trainer);
    void setSGD(float learningRate, float l2Reg, int numBatch, int numData, float momentum);
    void setRMSProp(float learningRate, float l2Reg, int numBatch, int numData, float decay);
    void setAdaGrad(float learningRate, float l2Reg, int numBatch, int numData);

    void updateParameters();

protected:
    Tensor<float> z;
    Tensor<float> weights;
    Tensor<float> biases;
    Tensor<float> output;
    Tensor<float> weight_delta; // dE/dw
    Tensor<float> bias_delta; // dE/db
    Tensor<float> delta; //dE/dz
    Tensor<float> dy_dz;

    int miniBatchSize;

    std::unique_ptr<TrainingRegime> trainer;
};

}
