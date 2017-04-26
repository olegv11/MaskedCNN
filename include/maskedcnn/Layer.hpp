#pragma once

#include <memory>
#include <string>
#include "Tensor.hpp"
#include "TrainingRegime.hpp"
namespace MaskedCNN
{

class Layer
{
public:
    Layer();
    Layer(Tensor<float>&& weights, Tensor<float>&& biases, std::string name = "");
    virtual ~Layer() = default;
    virtual void forwardPropagate() = 0;
    virtual void backwardPropagate() = 0;
    virtual std::vector<int> getOutputDimensions() = 0;
    virtual int getNeuronInputNumber() const { return 0; }
    void addBottom(Layer *layer);

    std::string getName() const;

    virtual const Tensor<float> *getOutput();
    virtual Tensor<float> *getDelta();

    void setTrainingMode(bool isTraining);

    void initializeWeightsStandardDistr();
    void initializeWeightsNormalDistrCorrectedVar();

    void setTrainer(std::unique_ptr<TrainingRegime> trainer);
    void setSGD(float learningRate, float l2Reg, int numBatch, int numData, float momentum);
    void setRMSProp(float learningRate, float l2Reg, int numBatch, int numData, float decay);
    void setAdaGrad(float learningRate, float l2Reg, int numBatch, int numData);

    void updateParameters();

protected:
    std::string name;
    Tensor<float> z;
    Tensor<float> weights;
    Tensor<float> biases;
    Tensor<float> output;
    Tensor<float> weight_delta; // dE/dw
    Tensor<float> bias_delta; // dE/db
    Tensor<float> delta; //dE/dz
    Tensor<float> dy_dz;

    int miniBatchSize;
    bool isTraining;
    bool initDone;

    std::unique_ptr<TrainingRegime> trainer;
    std::vector<Layer*> bottoms;
};

}
