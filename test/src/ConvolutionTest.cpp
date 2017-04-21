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
};

TEST_F(ConvolutionTest, SimpleConvolution) {
    Tensor<float> result(std::vector<int>{3,3});

    result(0,0) = 12; result(0,1) = 12; result(0,2) = 17;
    result(1,0) = 10; result(1,1) = 17; result(1,2) = 19;
    result(2,0) = 9;  result(2,1) = 6;  result(2,2) = 14;

    Tensor<float> output(std::vector<int>{1,3,3});

    convolution(matrix, weights, output, 3, 1, 0);

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

    convolution(matrix, weights, output, 3, 2, 1);

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

    convolution(input, filter, output, 1, 1, 0);

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

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  std::quick_exit(RUN_ALL_TESTS());
}
