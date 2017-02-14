#pragma once
#include <assert.h>
#include <vector>
#include <array>
#include <numeric>
#include <cblas.h>

namespace MaskedCNN
{

inline int outputConvDimension(int inDim, int filterDim, int stride, int pad)
{
    return (inDim - filterDim + 2 * pad) / stride + 1;
}

inline int multiplyAllElements(const std::vector<int> vec)
{
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<double>());
}

// [res] = [x]*[y]
inline void elementwiseMultiplication(const float *x, const float *y, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        res[i] = x[i] * y[i];
    }
}

inline void vectorCopy(float *__restrict__ dst, const float *__restrict__ src, int n)
{
    for (int i = 0; i < n; i++)
    {
        dst[i] = src[i];
    }
}


}
