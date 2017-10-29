#pragma once
#include "Tensor.hpp"
#include "Util.hpp"

namespace MaskedCNN {

int rot180(int f, int filterSize);

void convolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float>& out, int filterSize, int stride, int pad);
void transposedConvolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float>& out, int filterSize, int stride, int pad);
void im2col(const Tensor<float>& im, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& col);
int im2colMasked(const Tensor<float>& im, const Tensor<float>& mask, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& col);
void col2im(const Tensor<float>& col, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& im);
void convolutionIm2Col(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float>& out, int filterSize, int stride, int pad);
void transposedConvolutionIm2Col(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float>& out, int filterSize, int stride, int pad);
void convolutionIm2ColMasked(const Tensor<float>& input, const Tensor<float>& mask, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float> &outBuffer, Tensor<float>& out, int filterSize, int stride, int pad);
void convolutionIm2ColMaskedPlaceBufferBack(const Tensor<float>& mask, Tensor<float> &outBuffer, Tensor<float>& out);
void convolveMaskIm2Col(const Tensor<float>& prevMask, Tensor<float>& mask, Tensor<float>& colBuffer, int filterSize, int stride, int pad);
void deconvolveMaskCol2Im(const Tensor<float>& prevMask, Tensor<float>& mask, Tensor<float>& colBuffer, int filterSize, int stride, int pad);
void transposedConvolutionIm2ColMasked(const Tensor<float>& input, Tensor<float>& inputBuffer, const Tensor<float>& prevMask, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float>& anotherBuffer, Tensor<float>& out, int filterSize, int stride, int pad);
void col2imMasked(const Tensor<float>& col, const Tensor<float>& prevMask, int patches, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& im);



}
