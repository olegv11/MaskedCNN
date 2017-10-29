#include "Activation.hpp"
#include <cmath>
#include "../maskedcnncuda/ActivationCuda.h"
namespace MaskedCNN
{

void ReLu::activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = 0; i < num; i++)
    {
        y[i] = (x[i] > 0.0) ? x[i] : 0.0;
        delta[i] = (x[i] > 0.0) ? 1.0 : 0.0;
    }
}

void ReLu::activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    ReLu_activate_gpu(x,y,delta,num);
}

void Sigmoid::activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = 0; i < num; i++)
    {
        y[i] = 1.0 / (1.0 + std::exp(-x[i]));
        delta[i] = y[i] * (1 - y[i]);
    }
}

void Sigmoid::activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    Sigmoid_activate_gpu(x, y, delta, num);
}

void Tanh::activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = 0; i < num; i++)
    {
        float t = std::exp(2 * x[i]);
        y[i] = (t - 1) / (t + 1);
        delta[i] = (1 - y[i] * y[i]);
    }
}

void Tanh::activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    Tanh_activate_gpu(x, y, delta, num);
}

void Id::activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = 0; i < num; i++)
    {
        y[i] = x[i];
        delta[i] = 1;
    }
}

void Id::activate_gpu(const float *x, float *y, float *delta, int num)
{
    Id_activate_gpu(x, y, delta, num);
}

}


