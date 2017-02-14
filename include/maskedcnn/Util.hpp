#pragma once
#include <assert.h>
#include <vector>
#include <array>
#include <cblas.h>

#include "Tensor.hpp"
namespace MaskedCNN
{
    inline int outputConvDimension(int inDim, int filterDim, int stride, int pad)
    {
        return (inDim - filterDim + 2 * pad) / stride + 1;
    }
}
