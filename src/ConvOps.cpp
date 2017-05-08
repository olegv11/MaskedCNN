#include "Tensor.hpp"
#include "Util.hpp"

namespace MaskedCNN {



int rot180(int f, int filterSize)
{
    return filterSize - f - 1;
}

void convolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float>& out, int filterSize, int stride, int pad)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    out.zero();

    for (int ay = 0, y = -pad; ay < outputHeight; ay++, y += stride)
    {
        for (int ax = 0, x = -pad; ax < outputWidth; ax++, x += stride)
        {
            for (int fy = 0; fy < filterSize; fy++)
            {
                for (int fx = 0; fx < filterSize; fx++)
                {
                    int oy = y + fy;
                    int ox = x + fx;

                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                    {
                        for (int d = 0; d < outputChannels; d++)
                        {
                            for (int fd = 0; fd < inputChannels; fd++)
                            {
                                //std::cout << "out(" << d << "," << ay << "," << ax << ") "
                                //          << out(d, ay, ax) << "+=" << input(fd, oy, ox) << "*" << filter(d, fd, fy, fx) << std::endl;
                                out(d, ay, ax) += input(fd, oy, ox) * filter(d, fd, fy, fx);
                            }
                        }
                    }
                }
            }
        }
    }
}

void transposedConvolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float>& out, int filterSize, int stride, int pad)
{
    // Doing transposed convolution here
    // https://arxiv.org/pdf/1603.07285.pdf

    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    // "inserting" (stride-1) virtual zeroes in the transposed input between each "real" element
    const int transposedHeight = inputHeight + (inputHeight - 1) * (stride - 1);
    const int transposedWidth = inputWidth + (inputWidth - 1) * (stride - 1);

    const int virtualPad = filterSize - pad - 1;
    assert(virtualPad >= 0); // TODO: correctly compute it in cases where padding is more than full padding

    // Additional padding from left and up
    const int padLeft = (outputWidth + 2 * pad - filterSize) % stride;
    const int padUp = (outputHeight + 2 * pad - filterSize) % stride;

    const int shiftLeft = -(virtualPad + padLeft);
    const int shiftUp = -(virtualPad + padUp);

    out.zero();

    for (int ay = shiftUp; ay < transposedHeight + virtualPad - filterSize + 1; ay++)
    {
        for (int ax = shiftLeft; ax < transposedWidth + virtualPad - filterSize + 1; ax++)
        {
            for (int fy = 0; fy < filterSize; fy++)
            {
                int oy = ay + rot180(fy, filterSize);
                if ((oy % stride != 0) || oy < 0 || oy / stride >= inputHeight) continue;

                for (int fx = 0; fx < filterSize; fx++)
                {
                    int ox = ax + rot180(fx, filterSize);
                    if ((ox % stride != 0) || ox < 0 || ox / stride >= inputWidth) continue;

                    for (int d = 0; d < outputChannels; d++)
                    {
                        for (int fd = 0; fd < inputChannels; fd++)
                        {
                            /*std::cout << "out(" << d << "," << (ay - shiftUp)/ stride << "," << (ax  - shiftLeft)/ stride <<") "
                                      << out(d, (ay - shiftUp)/ stride, (ax  - shiftLeft)/ stride) << "+=input(" << fd <<","<<oy / stride<<","<<ox/stride<<") * filter("
                                      <<fd<<","<<d<<","<<fy<<","<<fx<<") "
                                      << input(fd, oy / stride, ox / stride) << "*" << filter(fd, d, fy, fx) << std::endl;*/
                            out(d, (ay - shiftUp), (ax  - shiftLeft)) += input(fd, oy / stride, ox / stride) * filter(fd, d, fy, fx);
                        }
                    }
                }
            }
        }
    }
    //std::cout << std::endl;
}

// Thanks to https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp for the reference implementation

void im2col(const Tensor<float>& im, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& col)
{
    auto dataCol = col.dataAddress();
    auto dataIm = im.dataAddress();
    const int outputHeight = (inputHeight + 2 * pad - filterSize) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - filterSize) / stride + 1;
    const int channelSize = inputHeight * inputWidth;

    for (int channel = 0; channel < inputChannels; dataIm += channelSize, channel++)
    {
        for (int fy = 0; fy < filterSize; fy++)
        {
            for (int fx = 0; fx < filterSize; fx++)
            {
                int y = -pad + fy;
                for (int outputRows = 0; outputRows < outputHeight; outputRows++)
                {
                    if (y < 0 || y >= inputHeight)
                    {
                        for (int outputCols = 0; outputCols < outputWidth; outputCols++)
                        {
                            *(dataCol++) = 0;
                        }
                    }
                    else
                    {
                        int x = -pad + fx;
                        for (int outputCols = 0; outputCols < outputWidth; outputCols++)
                        {
                            if (x >= 0 && x < inputWidth)
                            {
                                *(dataCol++) = im(channel, y, x);
                            }
                            else
                            {
                                *(dataCol++) = 0;
                            }
                            x += stride;
                        }
                    }
                    y += stride;
                }
            }
        }
    }
}

int im2colMasked(const Tensor<float>& im, const Tensor<float>& mask, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& col)
{
    auto dataCol = col.dataAddress();
    auto dataIm = im.dataAddress();
    auto dataMask = mask.dataAddress();
    const int outputHeight = (inputHeight + 2 * pad - filterSize) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - filterSize) / stride + 1;
    const int channelSize = inputHeight * inputWidth;

    int patchesToProcess = 0;

    for (int channel = 0; channel < inputChannels; dataIm += channelSize, channel++)
    {
        for (int fy = 0; fy < filterSize; fy++)
        {
            for (int fx = 0; fx < filterSize; fx++)
            {
                int y = -pad + fy;
                for (int outputRows = 0; outputRows < outputHeight; outputRows++)
                {
                    if (y < 0 || y >= inputHeight)
                    {
                        for (int outputCols = 0; outputCols < outputWidth; outputCols++)
                        {
                            if (dataMask[outputRows * outputWidth + outputCols] > 0)
                            {
                                patchesToProcess++;
                                *(dataCol++) = 0;
                            }
                        }
                    }
                    else
                    {
                        int x = -pad + fx;
                        for (int outputCols = 0; outputCols < outputWidth; outputCols++)
                        {
                            // Most important part: just skip if we don't need
                            // to compute (outputRows, outputCols)
                            if (dataMask[outputRows * outputWidth + outputCols] > 0)
                            {
                                patchesToProcess++;
                                if (x >= 0 && x < inputWidth)
                                {
                                    *(dataCol++) = im(channel, y, x);
                                }
                                else
                                {
                                    *(dataCol++) = 0;
                                }
                            }

                            x += stride;
                        }
                    }
                    y += stride;
                }
            }
        }
    }

    assert(patchesToProcess % inputChannels * filterSize * filterSize == 0);
    return patchesToProcess / (inputChannels * filterSize * filterSize);
}

void col2im(const Tensor<float>& col, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& im)
{
    im.zero();

    auto dataCol = col.dataAddress();
    auto dataIm = im.dataAddress();
    const int outputHeight = (inputHeight + 2 * pad - filterSize) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - filterSize) / stride + 1;
    const int channelSize = inputHeight * inputWidth;

    for (int channel = inputChannels; channel--; dataIm += channelSize)
    {
        for (int fy = 0; fy < filterSize; fy++)
        {
            for (int fx = 0; fx < filterSize; fx++)
            {
                int y = -pad + fy;
                for (int outputRows = outputHeight; outputRows; outputRows--)
                {
                    if (y < 0 || y >= inputHeight)
                    {
                        dataCol += outputWidth;
                    }
                    else
                    {
                        int x = -pad + fx;
                        for (int outputCols = outputWidth; outputCols; outputCols--)
                        {
                            if (x >= 0 && x < inputWidth)
                            {
                                dataIm[y * inputWidth + x] += *dataCol;
                            }
                            dataCol++;
                            x += stride;
                        }
                    }
                    y += stride;
                }
            }
        }
    }
}

void convolutionIm2Col(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float>& out, int filterSize, int stride, int pad)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    colBuffer.resize(std::vector<int>{inputChannels*filterSize*filterSize, outputHeight * outputWidth});

    im2col(input, inputChannels, inputHeight, inputWidth, filterSize, pad, stride, colBuffer);

    int m = outputChannels;
    int n = outputHeight * outputWidth;
    int k = inputChannels * filterSize * filterSize;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, filter.dataAddress(), k, colBuffer.dataAddress(),
                n, 0., out.dataAddress(), n);
}

void transposedConvolutionIm2Col(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float>& out, int filterSize, int stride, int pad)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    colBuffer.resize(std::vector<int>{outputChannels*filterSize*filterSize, inputHeight * inputWidth});

    int m = outputChannels * filterSize * filterSize;
    int n = inputHeight * inputWidth;
    int k = inputChannels;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k,
                1.0, filter.dataAddress(), m, input.dataAddress(),
                n, 0., colBuffer.dataAddress(), n);

    col2im(colBuffer, outputChannels, outputHeight, outputWidth, filterSize, pad, stride, out);
}

void convolutionIm2ColMasked(const Tensor<float>& input, const Tensor<float>& mask, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float> &outBuffer, Tensor<float>& out, int filterSize, int stride, int pad)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    // Always allocate for the worst case (have to process everything)
    colBuffer.resize(std::vector<int>{inputChannels*filterSize*filterSize, outputHeight * outputWidth});
    outBuffer.resize(out.dimensions());

    int patches = im2colMasked(input, mask, inputChannels, inputHeight, inputWidth, filterSize, pad, stride, colBuffer);

    int m = outputChannels;
    int n = patches;
    int k = inputChannels * filterSize * filterSize;

    auto outBufferData = outBuffer.dataAddress();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, filter.dataAddress(), k, colBuffer.dataAddress(),
                n, 0., outBufferData, n);
}

void convolutionIm2ColMaskedPlaceBufferBack(const Tensor<float>& mask, Tensor<float> &outBuffer, Tensor<float>& out)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];

    auto outBufferData = outBuffer.dataAddress();
    const auto& maskData = mask.dataAddress();
    const auto& outData = out.dataAddress();

    for (int c = 0; c < outputChannels; c++)
    {
        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                if (maskData[y * outputWidth + x] > 0)
                {
                    int index = x + outputWidth * (y + outputHeight * c);
                    outData[index] = *outBufferData;
                    outBufferData++;
                }
            }
        }
    }
}



void convolveMaskIm2Col(const Tensor<float>& prevMask, Tensor<float>& mask, Tensor<float>& colBuffer, int filterSize, int stride, int pad)
{
    const int outputHeight = mask.dimensions()[0];
    const int outputWidth = mask.dimensions()[1];
    const int inputHeight = prevMask.dimensions()[0];
    const int inputWidth = prevMask.dimensions()[1];

    mask.zero();

    Tensor<float> weights(std::vector<int>{1,1,filterSize,filterSize});
    weights.fillwith(1.0);

    const Tensor<float> deepPrevMask(prevMask, shallow_copy{});
    deepPrevMask.reshape(std::vector<int>{1, inputHeight, inputWidth});

    Tensor<float> deepMask(mask, shallow_copy{});
    deepMask.reshape(std::vector<int>{1, outputHeight, outputWidth});

    convolutionIm2Col(deepPrevMask, weights, colBuffer, deepMask, filterSize, stride, pad);
}



}

