#pragma once
#include "Layer.hpp"

#include <vector>

namespace MaskedCNN
{

class Network
{
public:
    Network();

private:
    std::vector<Layer> layers;
};

}
