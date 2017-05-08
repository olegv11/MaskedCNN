#pragma once
#include <string.h>
#include <math.h>
#include "Tensor.hpp"
namespace MaskedCNN
{

class TrainingRegime
{
public:
    TrainingRegime(float learningRate, float l2Reg, int numBatch, int numData)
        : numBatch(numBatch), numData(numData), learningRate(learningRate),
          l2Reg(l2Reg), counter(0), step(0)
    {
    }


    void updateParameters(float*__restrict__ w, float*__restrict__ dw, int numw,
                                  float*__restrict__ b, float*__restrict__ db, int numb)
    {
        if (!initDone)
        {
            this->numw = numw;
            this->numb = numb;
            partialdw.resize(numw);
            partialdb.resize(numb);
            memset(partialdw.data(), 0, numw * sizeof(float));
            memset(partialdb.data(), 0, numb * sizeof(float));
            initEnded();
            initDone = true;
        }
        else
        {
            assert(this->numw == numw);
            assert(this->numb == numb);
        }

        for(int i = 0; i < numw; i++)
        {
            partialdw[i] += dw[i];
        }

        for (int i = 0; i < numb; i++)
        {
            partialdb[i] += db[i];
        }

        counter++;

        if (counter % numBatch == 0)
        {
            step++;
            miniBatchEnded(w, b);
        }
    }

protected:
    virtual void miniBatchEnded(float*__restrict__ w, float*__restrict__ b) = 0;
    virtual void initEnded() {}

    std::vector<float> partialdw;
    std::vector<float> partialdb;

    int numw, numb, numBatch, numData;
    float learningRate;
    float l2Reg;

    int counter;
    int step;
    bool initDone = false;
};


class StochasticGradientDescent : public TrainingRegime
{
public:
    StochasticGradientDescent(float learningRate, float l2Reg, int numBatch, int numData, float momentum)
        :TrainingRegime(learningRate, l2Reg, numBatch, numData), momentum(momentum)
    {
    }

protected:
    virtual void miniBatchEnded(float*__restrict__ w, float*__restrict__ b) override
    {
        for (int i = 0; i < numw; i++)
        {
            float step = (partialdw[i] / numBatch) + l2Reg * w[i] / numData;
            w[i] -= learningRate * step + prevStepW[i] * momentum;
            prevStepW[i] =  step;
        }

        for (int i = 0; i < numb; i++)
        {
            float step = partialdb[i] / numBatch;
            b[i] -= learningRate * step + prevStepW[i] * momentum;
            prevStepB[i] = step;
        }

        memset(partialdw.data(), 0, numw * sizeof(float));
        memset(partialdb.data(), 0, numb * sizeof(float));
    }

    virtual void initEnded() override
    {
        prevStepW.resize(numw);
        prevStepB.resize(numb);
        memset(prevStepW.data(), 0, numw * sizeof(float));
        memset(prevStepB.data(), 0, numb * sizeof(float));
    }

    float momentum;
    std::vector<float> prevStepW;
    std::vector<float> prevStepB;
};

class AdaGrad : public TrainingRegime
{
public:
    AdaGrad(float learningRate, float l2Reg, int numBatch, int numData)
        :TrainingRegime(learningRate, l2Reg, numBatch, numData)
    {
    }

    virtual void miniBatchEnded(float*__restrict__ w, float*__restrict__ b) override
    {
        for (int i = 0; i < numw; i++)
        {
            float step = (partialdw[i] / numBatch) + l2Reg * w[i] / numData;
            sumGradW[i] += step * step;

            w[i] -= learningRate / (sqrt(sumGradW[i] + 1e-8)) * step;
        }

        for (int i = 0; i < numb; i++)
        {
            float step = partialdb[i] / numBatch;
            sumGradB[i] += step * step;

            b[i] -= learningRate / (sqrt(sumGradB[i] + 1e-8)) * step;
        }

        memset(partialdw.data(), 0, numw * sizeof(float));
        memset(partialdb.data(), 0, numb * sizeof(float));
    }

protected:
    virtual void initEnded() override
    {
        sumGradW.resize(numw);
        sumGradB.resize(numb);
        memset(sumGradW.data(), 0, numw * sizeof(float));
        memset(sumGradB.data(), 0, numb * sizeof(float));
    }

    std::vector<float> sumGradW;
    std::vector<float> sumGradB;
};

class RmsProp : public TrainingRegime
{
public:
    RmsProp(float learningRate, float l2Reg,int numBatch, int numData, float gamma)
        :TrainingRegime(learningRate, l2Reg, numBatch, numData), gamma(gamma)
    {
    }

    virtual void miniBatchEnded(float*__restrict__ w, float*__restrict__ b) override
    {
        for (int i = 0; i < numw; i++)
        {
            float step = (partialdw[i] / numBatch) + l2Reg * w[i] / numData;
            runningAverageW[i] = (1-gamma) * runningAverageW[i] + gamma * step * step;

            w[i] -= learningRate / (sqrt(runningAverageW[i] + 1e-8)) * step;
        }

        for (int i = 0; i < numb; i++)
        {
            float step = partialdb[i] / numBatch;
            runningAverageB[i] = (1-gamma) * runningAverageB[i] + gamma * step * step;

            b[i] -= learningRate / (sqrt(runningAverageB[i] + 1e-8)) * step;
        }

        memset(partialdw.data(), 0, numw * sizeof(float));
        memset(partialdb.data(), 0, numb * sizeof(float));
    }


protected:
    virtual void initEnded() override
    {
        runningAverageW.resize(numw);
        runningAverageB.resize(numb);
        memset(runningAverageW.data(), 0, numw * sizeof(float));
        memset(runningAverageB.data(), 0, numb * sizeof(float));
    }

    std::vector<float> runningAverageW;
    std::vector<float> runningAverageB;
    float gamma;
};

}

