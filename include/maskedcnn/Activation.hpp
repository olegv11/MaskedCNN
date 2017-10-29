#pragma once

namespace MaskedCNN
{

class Activation
{
public:
    virtual void activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) = 0;
    virtual void activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) = 0;
};


class ReLu : public Activation
{
public:
    virtual void activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
    virtual void activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
};

class Sigmoid : public Activation
{
public:
    virtual void activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
    virtual void activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
};

class Tanh : public Activation
{
public:
    virtual void activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
    virtual void activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
};

class Id : public Activation
{
public:
    virtual void activate(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
    virtual void activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num) override;
};

}
