#include "Activation.hpp"
#include <cmath>
namespace MaskedCNN
{

void ReLu::activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = 0; i < num; i++)
    {
        y[i] = x[i] > 0.0 ? x[i] : 0.0;
        delta[i] = x[i] > 0.0 ? 1.0 : 0.0;
    }
}

void Softmax::activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    float maxValue = x[0];
    for (int i = 1; i < num; i++)
    {
        if (x[i] > maxValue)
        {
            maxValue = x[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < num; i++)
    {
        y[i] = std::exp(x[i] - maxValue);
        sum += y[i];
    }

    sum = 1.0 / sum;

    for (int i = 0; i < num; i++)
    {
        y[i] *= sum;
        delta[i] = y[i] * (1 - y[i]);
    }
}

}


