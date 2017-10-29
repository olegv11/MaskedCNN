#include "NetworkLoader.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "fcntl.h"
#include <google/protobuf/io/coded_stream.h>

namespace MaskedCNN
{

void AddBottom(std::string bottom, std::vector<std::unique_ptr<Layer>>& result);

std::vector<std::unique_ptr<Layer>> loadCaffeNet(std::string path)
{
    caffe::NetParameter netParam;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        throw std::logic_error("File does not exist");
    }

    google::protobuf::io::FileInputStream fileInput(fd);
    fileInput.SetCloseOnDelete(true);

    google::protobuf::io::CodedInputStream code(&fileInput);
    code.SetTotalBytesLimit(1073741824, 536870912);

    if (!netParam.ParseFromCodedStream(&code))
    {
        throw std::logic_error("Could not parse");
    }


    std::vector<std::unique_ptr<Layer>> result;
    result.emplace_back(new InputLayer("data"));

    for (int i = 0; i < netParam.layers_size(); i++)
    {
        const caffe::V1LayerParameter& p = netParam.layers(i);
        std::string name = p.name();

        if (p.type() == caffe::V1LayerParameter_LayerType_CONVOLUTION)
        {
            const auto& weights = p.blobs(0);
            const auto& param = p.convolution_param();

            int oc = weights.num();
            int ic = weights.channels();
            int kh = weights.height();
            int kw = weights.width();

            Tensor<float> weightTensor(std::vector<int>{oc, ic, kh, kw});

            for (int outputChannel = 0; outputChannel < oc; outputChannel++)
            {
                for (int inputChannel = 0; inputChannel < ic; inputChannel++)
                {
                    for (int row = 0; row < kh; row++)
                    {
                        for (int col = 0; col < kw; col++)
                        {
                            weightTensor(outputChannel, inputChannel, row, col) =
                                    weights.data(col + kw * (row + kh * (inputChannel + ic * outputChannel)));
                        }
                    }
                }
            }

            Tensor<float> biasTensor(std::vector<int>{oc});
            if (p.blobs_size() >= 2)
            {
                const auto& biases = p.blobs(1);
                for (int outputChannel = 0; outputChannel < oc; outputChannel++)
                {
                    biasTensor[outputChannel] = biases.data(outputChannel);
                }
            }

            std::unique_ptr<Activation> act;
            if (netParam.layers(i+1).type() == caffe::V1LayerParameter_LayerType_RELU)
            {
                i++;
                act = std::make_unique<ReLu>();
            }
            else
            {
                act = std::make_unique<Id>();
            }

            int stride = param.stride_size() > 0 ? param.stride(0) : 1;
            int pad = param.pad_size() > 0 ? param.pad(0) : 0;

            result.emplace_back(new ConvolutionalLayer(std::move(act), std::move(weightTensor), std::move(biasTensor), stride, pad, name));
            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
        else if (p.type() == caffe::V1LayerParameter_LayerType_POOLING)
        {
            const auto& param = p.pooling_param();
            assert(param.pool() == caffe::PoolingParameter_PoolMethod_MAX);
            result.emplace_back(new PoolLayer(param.kernel_size(), name));

            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
        else if (p.type() == caffe::V1LayerParameter_LayerType_DROPOUT)
        {
            result.emplace_back(new DropoutLayer(p.dropout_param().dropout_ratio(), name));

            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
        else if (p.type() == caffe::V1LayerParameter_LayerType_SPLIT)
        {
            std::string bottom = p.bottom(0);
            for (int i = 0; i < p.top_size(); i++)
            {
                result.emplace_back(new PipeLayer(p.top(i)));
                AddBottom(bottom, result);
            }
        }
        else if (p.type() == caffe::V1LayerParameter_LayerType_INNER_PRODUCT)
        {
            const auto& weights = p.blobs(0);
            const auto& biases = p.blobs(1);

            int in = weights.width();
            int out = weights.height();


            Tensor<float> weightTensor(std::vector<int>{out, in});

            for (int y = 0; y < out; y++)
            {
                for (int x = 0; x < in; x++)
                {
                    weightTensor(y,x) = weights.data(x + y * in);
                }
            }


            Tensor<float> biasTensor(std::vector<int>{out});
            for (int x = 0; x < out; x++)
            {
                biasTensor[x] = biases.data(x);
            }

            std::unique_ptr<Activation> act;
            if (netParam.layers(i+1).type() == caffe::V1LayerParameter_LayerType_RELU)
            {
                i++;
                act = std::make_unique<ReLu>();
            }
            else
            {
                act = std::make_unique<Id>();
            }

            result.emplace_back(new FullyConnectedLayer(std::move(act), std::move(weightTensor), std::move(biasTensor), name));

            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
    }

    for (int i = 0; i < netParam.layer_size(); i++)
    {
        const caffe::LayerParameter& p = netParam.layer(i);
        std::string name = p.name();
        if (p.type() == "Convolution" || p.type() == "Deconvolution")
        {
            const auto& weights = p.blobs(0);
            const auto& param = p.convolution_param();

            int oc = weights.shape().dim(0);
            int ic = weights.shape().dim(1);
            int kh = weights.shape().dim(2);
            int kw = weights.shape().dim(3);


            Tensor<float> weightTensor(std::vector<int>{oc, ic, kh, kw});

            for (int outputChannel = 0; outputChannel < oc; outputChannel++)
            {
                for (int inputChannel = 0; inputChannel < ic; inputChannel++)
                {
                    for (int row = 0; row < kh; row++)
                    {
                        for (int col = 0; col < kw; col++)
                        {
                            weightTensor(outputChannel, inputChannel, row, col) =
                                    weights.data(col + kw * (row + kh * (inputChannel + ic * outputChannel)));
                        }
                    }
                }
            }

            Tensor<float> biasTensor(std::vector<int>{oc});
            if (p.blobs_size() >= 2)
            {
                const auto& biases = p.blobs(1);
                for (int outputChannel = 0; outputChannel < oc; outputChannel++)
                {
                    biasTensor[outputChannel] = biases.data(outputChannel);
                }
            }

            std::unique_ptr<Activation> act;
            if (netParam.layer(i+1).type() == "ReLU")
            {
                i++;
                act = std::make_unique<ReLu>();
            }
            else
            {
                act = std::make_unique<Id>();
            }

            int stride = param.stride_size() > 0 ? param.stride(0) : 1;
            int pad = param.pad_size() > 0 ? param.pad(0) : 0;

            if (p.type() == "Convolution")
            {
                result.emplace_back(new ConvolutionalLayer(std::move(act), std::move(weightTensor), std::move(biasTensor), stride, pad, name));
            }
            else
            {
                result.emplace_back(new DeconvolutionalLayer(std::move(act), std::move(weightTensor), std::move(biasTensor), stride, pad, name));
            }

            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
        else if (p.type() == "Pooling")
        {
            const auto& param = p.pooling_param();
            assert(param.pool() == caffe::PoolingParameter_PoolMethod_MAX);
            result.emplace_back(new PoolLayer(param.kernel_size(), name));

            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
        else if (p.type() == "Dropout")
        {
            result.emplace_back(new DropoutLayer(p.dropout_param().dropout_ratio(), name));

            std::string bottom = p.bottom(0);
            AddBottom(bottom, result);
        }
        else if (p.type() == "Split")
        {
            std::string bottom = p.bottom(0);
            for (int i = 0; i < p.top_size(); i++)
            {
                result.emplace_back(new PipeLayer(p.top(i)));
                AddBottom(bottom, result);
            }
        }
        else if (p.type() == "Eltwise")
        {

        }
    }

    return result;
}

void AddBottom(std::string bottom, std::vector<std::unique_ptr<Layer>>& result)
{
    auto bottomLayer = std::find_if(result.begin(), result.end(), [&](auto& l){return l->getName() == bottom;});
    if (bottomLayer == result.end())
    {
        throw std::exception();
    }
    result[result.size() - 1]->addBottom(bottomLayer->get());
}

}

