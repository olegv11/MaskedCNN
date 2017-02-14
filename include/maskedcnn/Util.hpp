#pragma once
#include <assert.h>
#include <vector>
#include <array>
#include <numeric>
#include <cblas.h>

namespace MaskedCNN
{

int outputConvDimension(int inDim, int filterDim, int stride, int pad);
int multiplyAllElements(const std::vector<int> vec);

}
