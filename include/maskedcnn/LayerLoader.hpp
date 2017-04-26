#pragma once

#include <string>
#include <vector>

#include "Layer.hpp"
#include "ConvolutionalLayer.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "DropoutLayer.hpp"
#include "PoolLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "PipeLayer.hpp"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "caffe.pb.h"

namespace MaskedCNN {

std::vector<std::unique_ptr<Layer>> loadCaffeNet(std::string path, int width, int height, int channels);


}
