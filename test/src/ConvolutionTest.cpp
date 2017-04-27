#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include "ConvOps.hpp"
#include "Tensor.hpp"

using namespace MaskedCNN;

namespace {

// The fixture for testing class Foo.
class ConvolutionTest : public ::testing::Test {
protected:
    ConvolutionTest()
        :weights(std::vector<int>{1,1,3,3}), matrix(std::vector<int>{1,5,5}), delta(std::vector<int>({1,3,3}))
    {
    }

    virtual void SetUp()
    {
        weights(0,0,0,0) = 0; weights(0,0,0,1) = 1; weights(0,0,0,2) = 2;
        weights(0,0,1,0) = 2; weights(0,0,1,1) = 2; weights(0,0,1,2) = 0;
        weights(0,0,2,0) = 0; weights(0,0,2,1) = 1; weights(0,0,2,2) = 2;

        matrix(0,0,0) = 3; matrix(0,0,1) = 3; matrix(0,0,2) = 2; matrix(0,0,3) = 1; matrix(0,0,4) = 0;
        matrix(0,1,0) = 0; matrix(0,1,1) = 0; matrix(0,1,2) = 1; matrix(0,1,3) = 3; matrix(0,1,4) = 1;
        matrix(0,2,0) = 3; matrix(0,2,1) = 1; matrix(0,2,2) = 2; matrix(0,2,3) = 2; matrix(0,2,4) = 3;
        matrix(0,3,0) = 2; matrix(0,3,1) = 0; matrix(0,3,2) = 0; matrix(0,3,3) = 2; matrix(0,3,4) = 2;
        matrix(0,4,0) = 2; matrix(0,4,1) = 0; matrix(0,4,2) = 0; matrix(0,4,3) = 0; matrix(0,4,4) = 1;

        delta(0,0,0) = 6; delta(0,0,1) = 14; delta(0,0,2) = 17;
        delta(0,1,0) = 14; delta(0,1,1) = 12; delta(0,1,2) = 12;
        delta(0,2,0) = 8; delta(0,2,1) = 10; delta(0,2,2) = 17;
    }

    Tensor<float> weights;
    Tensor<float> matrix;
    Tensor<float> delta;
    Tensor<float> colBuffer;
};

TEST_F(ConvolutionTest, SimpleConvolution) {
    Tensor<float> result(std::vector<int>{3,3});

    result(0,0) = 12; result(0,1) = 12; result(0,2) = 17;
    result(1,0) = 10; result(1,1) = 17; result(1,2) = 19;
    result(2,0) = 9;  result(2,1) = 6;  result(2,2) = 14;

    Tensor<float> output(std::vector<int>{1,3,3});

    convolutionIm2Col(matrix, weights, colBuffer, output, 3, 1, 0);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ASSERT_FLOAT_EQ(result(i,j), output(0,i,j));
        }
    }
}

TEST_F(ConvolutionTest, PaddedStridedConvolution) {
    Tensor<float> result(std::vector<int>{3,3});

    result(0,0) = 6; result(0,1) = 17; result(0,2) = 3;
    result(1,0) = 8; result(1,1) = 17; result(1,2) = 13;
    result(2,0) = 6;  result(2,1) = 4;  result(2,2) = 4;

    Tensor<float> output(std::vector<int>{1,3,3});

    convolutionIm2Col(matrix, weights, colBuffer, output, 3, 2, 1);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ASSERT_FLOAT_EQ(result(i,j), output(0,i,j));
        }
    }
}

TEST_F(ConvolutionTest, MultidimensionalSimpleConvolution) {
    Tensor<float> input(std::vector<int>{3,1,1});

    input(0,0,0) = 1;
    input(1,0,0) = 7;
    input(2,0,0) = 9;

    Tensor<float> filter(std::vector<int>{1,3,1,1});
    filter(0,0,0,0) = 5;
    filter(0,1,0,0) = 2;
    filter(0,2,0,0) = 13;

    Tensor<float> output(std::vector<int>{1,1,1});

    convolutionIm2Col(input, filter, colBuffer, output, 1, 1, 0);

    ASSERT_FLOAT_EQ(output(0,0,0), 136);
}

TEST_F(ConvolutionTest, SimpleTransposedConvolution) {
    Tensor<float> result(std::vector<int>{5,5});
    result(0,0) = 0; result(0,1) = 6; result(0,2) = 26; result(0,3) = 45; result(0,4) = 34;
    result(1,0) = 12; result(1,1) = 54; result(1,2) = 102; result(1,3) = 70; result(1,4) = 24;
    result(2,0) = 28; result(2,1) = 66; result(2,2) = 100; result(2,3) = 106; result(2,4) = 68;
    result(3,0) = 16; result(3,1) = 50; result(3,2) = 94; result(3,3) = 70; result(3,4) = 24;
    result(4,0) = 0; result(4,1) = 8; result(4,2) = 26; result(4,3) = 37; result(4,4) = 34;

    Tensor<float> output(std::vector<int>{1,5,5});


    transposedConvolution(delta, weights, output, 3, 1, 0);

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            ASSERT_FLOAT_EQ(result(i,j), output(0,i,j));
        }
    }
}

TEST_F(ConvolutionTest, SameTransposedConvolution) {
    Tensor<float> result(std::vector<int>{3,3});
    result(0,0) = 54; result(0,1) = 102; result(0,2) = 70;
    result(1,0) = 66; result(1,1) = 100; result(1,2) = 106;
    result(2,0) = 50; result(2,1) = 94; result(2,2) = 70;


    Tensor<float> output(std::vector<int>{1,5,5});


    transposedConvolution(delta, weights, output, 3, 1, 1);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ASSERT_FLOAT_EQ(result(i,j), output(0,i,j));
        }
    }
}

TEST_F(ConvolutionTest, FullPaddingTransposedConvolution) {
    Tensor<float> output(std::vector<int>{1,1,1});
    Tensor<float> input(std::vector<int>{1,2,2});

    input(0,0,0) = 10; input(0,0,1) = 2;
    input(0,1,0) = 3;  input(0,1,1) = 7;

    transposedConvolution(input, weights, output, 3, 2, 2);

    ASSERT_FLOAT_EQ(output(0,0,0), 26);
}

TEST_F(ConvolutionTest, MultiChannelConvolution)
{
    Tensor<float> input(std::vector<int>{2,3,3});
    Tensor<float> weights(std::vector<int>{2,1,3,3});
    Tensor<float> result(std::vector<int>{1,5,5});
    Tensor<float> output(std::vector<int>{1,5,5});

    input(0,0,0) = 6; input(0,0,1) = 14; input(0,0,2) = 17;
    input(0,1,0) = 14; input(0,1,1) = 12; input(0,1,2) = 12;
    input(0,2,0) = 8; input(0,2,1) = 10; input(0,2,2) = 17;

    input(1,0,0) = 1; input(1,0,1) = 2; input(1,0,2) = 3;
    input(1,1,0) = 4; input(1,1,1) = 5; input(1,1,2) = 6;
    input(1,2,0) = 7; input(1,2,1) = 8; input(1,2,2) = 9;


    weights(0,0,0,0) = 0; weights(0,0,0,1) = 1; weights(0,0,0,2) = 2;
    weights(0,0,1,0) = 2; weights(0,0,1,1) = 2; weights(0,0,1,2) = 0;
    weights(0,0,2,0) = 0; weights(0,0,2,1) = 1; weights(0,0,2,2) = 1;

    weights(1,0,0,0) = 3; weights(1,0,0,1) = 4; weights(1,0,0,2) = 10;
    weights(1,0,1,0) = 5; weights(1,0,1,1) = 1; weights(1,0,1,2) = 1;
    weights(1,0,2,0) = 3; weights(1,0,2,1) = 0; weights(1,0,2,2) = 7;


    result(0,0,0) = 3; result(0,0,1) = 16; result(0,0,2) = 53; result(0,0,3) = 77; result(0,0,4) = 64;
    result(0,1,0) = 29; result(0,1,1) = 96; result(0,1,2) = 198; result(0,1,3) = 149; result(0,1,4) = 87;
    result(0,2,0) = 72; result(0,2,1) = 153; result(0,2,2) = 278; result(0,2,3) = 233; result(0,2,4) = 168;
    result(0,3,0) = 63; result(0,3,1) = 112; result(0,3,2) = 186; result(0,3,3) = 110; result(0,3,4) = 63;
    result(0,4,0) = 21; result(0,4,1) = 32; result(0,4,2) = 94; result(0,4,3) = 83; result(0,4,4) = 80;

    transposedConvolution(input, weights, output, 3, 1, 0);

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            ASSERT_FLOAT_EQ(result(0,i,j), output(0,i,j));
        }
    }
}

TEST_F(ConvolutionTest, BigConvolution)
{
    Tensor<float> input(std::vector<int>{1,7,7});
    Tensor<float> weights(std::vector<int>{1,1,3,3});
    Tensor<float> result(std::vector<int>{1,9,9});
    Tensor<float> output(std::vector<int>{1,9,9});

    input(0,0,0) = 1; input(0,0,1) = 2; input(0,0,2) = 7; input(0,0,3) = 5; input(0,0,4) = 8; input(0,0,5) = 10; input(0,0,6) = 2;
    input(0,1,0) = 2; input(0,1,1) = 0; input(0,1,2) = 3; input(0,1,3) = 4; input(0,1,4) = 7; input(0,1,5) = 11; input(0,1,6) = 5;
    input(0,2,0) = 7; input(0,2,1) = 9; input(0,2,2) = 8; input(0,2,3) = 11; input(0,2,4) = 3; input(0,2,5) = 0; input(0,2,6) = 2;
    input(0,3,0) = 8; input(0,3,1) = 7; input(0,3,2) = 9; input(0,3,3) = 1; input(0,3,4) = 5; input(0,3,5) = 3; input(0,3,6) = 4;
    input(0,4,0) = 2; input(0,4,1) = 5; input(0,4,2) = 3; input(0,4,3) = 0; input(0,4,4) = 5; input(0,4,5) = 0; input(0,4,6) = 7;
    input(0,5,0) = 0; input(0,5,1) = 7; input(0,5,2) = 2; input(0,5,3) = 5; input(0,5,4) = 3; input(0,5,5) = 6; input(0,5,6) = 7;
    input(0,6,0) = 1; input(0,6,1) = 8; input(0,6,2) = 8; input(0,6,3) = 7; input(0,6,4) = 0; input(0,6,5) = 0; input(0,6,6) = 3;

    weights(0,0,0,0) = 0; weights(0,0,0,1) = 1; weights(0,0,0,2) = 2;
    weights(0,0,1,0) = 2; weights(0,0,1,1) = 2; weights(0,0,1,2) = 0;
    weights(0,0,2,0) = 0; weights(0,0,2,1) = 1; weights(0,0,2,2) = 1;


    result(0,0,0) = 0; result(0,0,1) = 1; result(0,0,2) = 4; result(0,0,3) = 11; result(0,0,4) = 19; result(0,0,5) = 18; result(0,0,6) = 26; result(0,0,7) = 22; result(0,0,8) = 4;
    result(0,1,0) = 2; result(0,1,1) = 8; result(0,1,2) = 22; result(0,1,3) = 27; result(0,1,4) = 36; result(0,1,5) = 51; result(0,1,6) = 49; result(0,1,7) = 31; result(0,1,8) = 10;
    result(0,2,0) = 4; result(0,2,1) = 12; result(0,2,2) = 32; result(0,2,3) = 49; result(0,2,4) = 61; result(0,2,5) = 74; result(0,2,6) = 56; result(0,2,7) = 24; result(0,2,8) = 6;
    result(0,3,0) = 14; result(0,3,1) = 42; result(0,3,2) = 59; result(0,3,3) = 64; result(0,3,4) = 54; result(0,3,5) = 24; result(0,3,6) = 35; result(0,3,7) = 30; result(0,3,8) = 13;
    result(0,4,0) = 16; result(0,4,1) = 39; result(0,4,2) = 57; result(0,4,3) = 50; result(0,4,4) = 37; result(0,4,5) = 35; result(0,4,6) = 27; result(0,4,7) = 17; result(0,4,8) = 16;
    result(0,5,0) = 4; result(0,5,1) = 22; result(0,5,2) = 38; result(0,5,3) = 38; result(0,5,4) = 29; result(0,5,5) = 29; result(0,5,6) = 34; result(0,5,7) = 40; result(0,5,8) = 18;
    result(0,6,0) = 0; result(0,6,1) = 17; result(0,6,2) = 35; result(0,6,3) = 46; result(0,6,4) = 42; result(0,6,5) = 37; result(0,6,6) = 31; result(0,6,7) = 24; result(0,6,8) = 13;
    result(0,7,0) = 2; result(0,7,1) = 18; result(0,7,2) = 39; result(0,7,3) = 39; result(0,7,4) = 21; result(0,7,5) = 8; result(0,7,6) = 15; result(0,7,7) = 19; result(0,7,8) = 7;
    result(0,8,0) = 0; result(0,8,1) = 1; result(0,8,2) = 9; result(0,8,3) = 16; result(0,8,4) = 15; result(0,8,5) = 7; result(0,8,6) = 0; result(0,8,7) = 3; result(0,8,8) = 3;

    transposedConvolution(input, weights, output, 3, 1, 0);

    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            ASSERT_FLOAT_EQ(output(0,i,j), result(0,i,j));
        }
    }
}

TEST_F(ConvolutionTest, BigStridedConvolution)
{
    Tensor<float> input(std::vector<int>{1,3,3});
    Tensor<float> weights(std::vector<int>{1,1,3,3});
    Tensor<float> result(std::vector<int>{1,15,15});
    Tensor<float> output(std::vector<int>{1,15,15});

    input(0,0,0) = 1; input(0,0,1) = 2; input(0,0,2) = 3;
    input(0,1,0) = 4; input(0,1,1) = 5; input(0,1,2) = 6;
    input(0,2,0) = 7; input(0,2,1) = 8; input(0,2,2) = 9;

    weights(0,0,0,0) = 0; weights(0,0,0,1) = 1; weights(0,0,0,2) = 2;
    weights(0,0,1,0) = 2; weights(0,0,1,1) = 2; weights(0,0,1,2) = 0;
    weights(0,0,2,0) = 0; weights(0,0,2,1) = 1; weights(0,0,2,2) = 1;


    result(0,0,0) = 0; result(0,0,1) = 1; result(0,0,2) = 2; result(0,0,3) = 0; result(0,0,4) = 0; result(0,0,5) = 0; result(0,0,6) = 0; result(0,0,7) = 2; result(0,0,8) = 4; result(0,0,9) = 0; result(0,0,10) = 0; result(0,0,11) = 0; result(0,0,12) = 0; result(0,0,13) = 3; result(0,0,14) = 6;
    result(0,1,0) = 2; result(0,1,1) = 2; result(0,1,2) = 0; result(0,1,3) = 0; result(0,1,4) = 0; result(0,1,5) = 0; result(0,1,6) = 4; result(0,1,7) = 4; result(0,1,8) = 0; result(0,1,9) = 0; result(0,1,10) = 0; result(0,1,11) = 0; result(0,1,12) = 6; result(0,1,13) = 6; result(0,1,14) = 0;
    result(0,2,0) = 0; result(0,2,1) = 1; result(0,2,2) = 1; result(0,2,3) = 0; result(0,2,4) = 0; result(0,2,5) = 0; result(0,2,6) = 0; result(0,2,7) = 2; result(0,2,8) = 2; result(0,2,9) = 0; result(0,2,10) = 0; result(0,2,11) = 0; result(0,2,12) = 0; result(0,2,13) = 3; result(0,2,14) = 3;
    result(0,3,0) = 0; result(0,3,1) = 0; result(0,3,2) = 0; result(0,3,3) = 0; result(0,3,4) = 0; result(0,3,5) = 0; result(0,3,6) = 0; result(0,3,7) = 0; result(0,3,8) = 0; result(0,3,9) = 0; result(0,3,10) = 0; result(0,3,11) = 0; result(0,3,12) = 0; result(0,3,13) = 0; result(0,3,14) = 0;
    result(0,4,0) = 0; result(0,4,1) = 0; result(0,4,2) = 0; result(0,4,3) = 0; result(0,4,4) = 0; result(0,4,5) = 0; result(0,4,6) = 0; result(0,4,7) = 0; result(0,4,8) = 0; result(0,4,9) = 0; result(0,4,10) = 0; result(0,4,11) = 0; result(0,4,12) = 0; result(0,4,13) = 0; result(0,4,14) = 0;
    result(0,5,0) = 0; result(0,5,1) = 0; result(0,5,2) = 0; result(0,5,3) = 0; result(0,5,4) = 0; result(0,5,5) = 0; result(0,5,6) = 0; result(0,5,7) = 0; result(0,5,8) = 0; result(0,5,9) = 0; result(0,5,10) = 0; result(0,5,11) = 0; result(0,5,12) = 0; result(0,5,13) = 0; result(0,5,14) = 0;
    result(0,6,0) = 0; result(0,6,1) = 4; result(0,6,2) = 8; result(0,6,3) = 0; result(0,6,4) = 0; result(0,6,5) = 0; result(0,6,6) = 0; result(0,6,7) = 5; result(0,6,8) = 10; result(0,6,9) = 0; result(0,6,10) = 0; result(0,6,11) = 0; result(0,6,12) = 0; result(0,6,13) = 6; result(0,6,14) = 12;
    result(0,7,0) = 8; result(0,7,1) = 8; result(0,7,2) = 0; result(0,7,3) = 0; result(0,7,4) = 0; result(0,7,5) = 0; result(0,7,6) = 10; result(0,7,7) = 10; result(0,7,8) = 0; result(0,7,9) = 0; result(0,7,10) = 0; result(0,7,11) = 0; result(0,7,12) = 12; result(0,7,13) = 12; result(0,7,14) = 0;
    result(0,8,0) = 0; result(0,8,1) = 4; result(0,8,2) = 4; result(0,8,3) = 0; result(0,8,4) = 0; result(0,8,5) = 0; result(0,8,6) = 0; result(0,8,7) = 5; result(0,8,8) = 5; result(0,8,9) = 0; result(0,8,10) = 0; result(0,8,11) = 0; result(0,8,12) = 0; result(0,8,13) = 6; result(0,8,14) = 6;
    result(0,9,0) = 0; result(0,9,1) = 0; result(0,9,2) = 0; result(0,9,3) = 0; result(0,9,4) = 0; result(0,9,5) = 0; result(0,9,6) = 0; result(0,9,7) = 0; result(0,9,8) = 0; result(0,9,9) = 0; result(0,9,10) = 0; result(0,9,11) = 0; result(0,9,12) = 0; result(0,9,13) = 0; result(0,9,14) = 0;
    result(0,10,0) = 0; result(0,10,1) = 0; result(0,10,2) = 0; result(0,10,3) = 0; result(0,10,4) = 0; result(0,10,5) = 0; result(0,10,6) = 0; result(0,10,7) = 0; result(0,10,8) = 0; result(0,10,9) = 0; result(0,10,10) = 0; result(0,10,11) = 0; result(0,10,12) = 0; result(0,10,13) = 0; result(0,10,14) = 0;
    result(0,11,0) = 0; result(0,11,1) = 0; result(0,11,2) = 0; result(0,11,3) = 0; result(0,11,4) = 0; result(0,11,5) = 0; result(0,11,6) = 0; result(0,11,7) = 0; result(0,11,8) = 0; result(0,11,9) = 0; result(0,11,10) = 0; result(0,11,11) = 0; result(0,11,12) = 0; result(0,11,13) = 0; result(0,11,14) = 0;
    result(0,12,0) = 0; result(0,12,1) = 7; result(0,12,2) = 14; result(0,12,3) = 0; result(0,12,4) = 0; result(0,12,5) = 0; result(0,12,6) = 0; result(0,12,7) = 8; result(0,12,8) = 16; result(0,12,9) = 0; result(0,12,10) = 0; result(0,12,11) = 0; result(0,12,12) = 0; result(0,12,13) = 9; result(0,12,14) = 18;
    result(0,13,0) = 14; result(0,13,1) = 14; result(0,13,2) = 0; result(0,13,3) = 0; result(0,13,4) = 0; result(0,13,5) = 0; result(0,13,6) = 16; result(0,13,7) = 16; result(0,13,8) = 0; result(0,13,9) = 0; result(0,13,10) = 0; result(0,13,11) = 0; result(0,13,12) = 18; result(0,13,13) = 18; result(0,13,14) = 0;
    result(0,14,0) = 0; result(0,14,1) = 7; result(0,14,2) = 7; result(0,14,3) = 0; result(0,14,4) = 0; result(0,14,5) = 0; result(0,14,6) = 0; result(0,14,7) = 8; result(0,14,8) = 8; result(0,14,9) = 0; result(0,14,10) = 0; result(0,14,11) = 0; result(0,14,12) = 0; result(0,14,13) = 9; result(0,14,14) = 9;

    transposedConvolution(input, weights, output, 3, 6, 0);

    for (int i = 0; i < 15; i++)
    {
        for (int j = 0; j < 15; j++)
        {
            ASSERT_FLOAT_EQ(output(0,i,j), result(0,i,j));
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  std::quick_exit(RUN_ALL_TESTS());
}
