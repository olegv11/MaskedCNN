#pragma once
#include <vector>
#include <functional>
#include <numeric>
#include <cstring>
#include "Util.hpp"
#include <iostream>
namespace MaskedCNN
{

// Better to make it explicit
class shallow_copy {};


// Row-major order
template<typename T>
class Tensor
{
public:
    Tensor();
    Tensor(std::vector<int> &dimensions);
    Tensor(std::vector<int> &&dimensions);
    Tensor(int channelLength, int columnLength, int rowLength);
    Tensor(int channelLength2, int channelLength, int columnLength, int rowLength);
    Tensor(const Tensor<T>& other); // deep copy
    Tensor(const Tensor<T>& other, shallow_copy) noexcept;
    Tensor(Tensor<T>&& other) noexcept;
    Tensor<T>& operator=(const Tensor<T>& other);
    Tensor<T>& operator=(Tensor<T>&& other) noexcept;
    ~Tensor();

    int rowLength() const { return dims[dimensionCount() - 1]; }
    int columnLength() const { return dims[dimensionCount() - 2]; }
    int channelLength() const { return dims[dimensionCount() - 3]; }
    int channel2Length() const { return dims[dimensionCount() - 4]; }

    float* dataAddress();

    T& operator[](size_t index);
    const T& operator[](size_t index) const;


    T& operator()(int row, int column);
    const T& operator()(int row, int column) const;

    T& operator()(int channel, int row, int column);
    const T& operator()(int channel, int row, int column) const;

    T& operator()(int channel2, int channel, int row, int column);
    const T& operator()(int channel2, int channel, int row, int column) const;

    bool sameShape(const Tensor& other) const;

    void reshape(std::vector<int> &dimensions);
    void reshape(int rowLength, int columnLength, int channelLength);
    void resize(const std::vector<int> &dimensions);
    void flatten();
    void fillwith(T scalar);

    void zero();

    int elementCount() const;
    std::vector<int> dimensions() const;
    int dimensionCount() const;

    Tensor<T> pad(int horizontalPad, int verticalPad, T padValue) const;
    Tensor<T> unpad(int horizontalPad, int verticalPad) const;

    double mean();
    T max();


private:
    std::vector<int> dims;
    T *data;
    bool isShallow;
};

template<typename T>
Tensor<T>::Tensor()
    :data(nullptr), isShallow(false)
{
}

template<typename T>
Tensor<T>::Tensor(std::vector<int> &dimensions)
    :dims(dimensions), isShallow(false)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(std::vector<int> &&dimensions)
    :dims(std::move(dimensions)), isShallow(false)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(int channelLength, int columnLength, int rowLength)
    :dims{channelLength, columnLength, rowLength}, isShallow(false)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(int channelLength2, int channelLength, int columnLength, int rowLength)
    :dims{channelLength2, channelLength, columnLength, rowLength}, isShallow(false)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other)
    :dims(other.dims), isShallow(false)
{
    data = new T[other.elementCount()];
    std::memcpy(data, other.data, other.elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other, shallow_copy) noexcept
    :dims(other.dims), data(other.data), isShallow(true)
{
}

template<typename T>
Tensor<T>::Tensor(Tensor<T> &&other) noexcept
    :dims(std::move(other.dims)), data(std::move(other.data))
{
    isShallow = other.isShallow;
    other.data = nullptr;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other)
{
    if (this == &other)
    {
        return *this;
    }

    if (elementCount() == other.elementCount())
    {
        dims = other.dims;
        std::memcpy(data, other.data, elementCount() * sizeof(T));
    }
    else
    {
        delete[] data;
        dims = other.dims;
        data = new T[elementCount()];
    }

    isShallow = false;
    return *this;

}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept
{
    dims = std::move(other.data);
    data = std::move(other.data);

    isShallow = other.isShallow;
    other.data = nullptr;

    return *this;
}

template<typename T>
Tensor<T>::~Tensor()
{
    if (!isShallow)
    {
        delete[] data;
    }
}

template<typename T>
float* Tensor<T>::dataAddress()
{
    return data;
}

template<typename T>
T& Tensor<T>::operator[](size_t index)
{
    return data[index];
}

template<typename T>
const T& Tensor<T>::operator[](size_t index) const
{
    return data[index];
}

template<typename T>
T& Tensor<T>::operator()(int row, int column)
{
    assert(dimensionCount() == 2);
    assert(column < this->dims[1]);
    assert(row < this->dims[0]);
    return data[row * dims[1] + column];
}

template<typename T>
const T& Tensor<T>::operator()(int row, int column) const
{
    assert(dimensionCount() == 2);
    assert(column < this->dims[1]);
    assert(row < this->dims[0]);
    return data[row * dims[1] + column];
}

template<typename T>
T& Tensor<T>::operator()(int channel, int row, int column)
{
    assert(dimensionCount() == 3);
    assert(column < this->dims[2]);
    assert(row < this->dims[1]);
    assert(channel < this->dims[0]);
    return data[channel * dims[2] * dims[1] + row * dims[2] + column];
}

template<typename T>
const T& Tensor<T>::operator()(int channel, int row, int column) const
{
   assert(dimensionCount() == 3);
   assert(row < this->dims[2]);
   assert(column < this->dims[1]);
   assert(channel < this->dims[0]);
   return data[(channel * dims[1] + column) * dims[2] + row];
}

template<typename T>
T& Tensor<T>::operator()(int channel2, int channel, int row, int column)
{
    assert(dimensionCount() == 4);
    assert(column < this->dims[3]);
    assert(row < this->dims[2]);
    assert(channel < this->dims[1]);
    assert(channel2 < this->dims[0]);
    return data[((channel2 * dims[1] + channel) * dims[2] + row) * dims[3] + column];
}

template<typename T>
const T& Tensor<T>::operator()(int channel2, int channel, int row, int column) const
{
    assert(dimensionCount() == 4);
    assert(column < this->dims[3]);
    assert(row < this->dims[2]);
    assert(channel < this->dims[1]);
    assert(channel2 < this->dims[0]);
    return data[((channel2 * dims[1] + channel) * dims[2] + row) * dims[3] + column];
}


template<typename T>
bool Tensor<T>::sameShape(const Tensor &other) const
{
    return dims == other.dims;
}

template<typename T>
void Tensor<T>::reshape(std::vector<int> &dimensions)
{
    int newElementCount = multiplyAllElements(dimensions);
    if (newElementCount != elementCount())
    {
        throw std::runtime_error("Invalid reshape");
    }

    dims = dimensions;
}

template<typename T>
void Tensor<T>::reshape(int width, int height, int channels)
{
    int newElementCount = width * height * channels;
    if (newElementCount != elementCount())
    {
        throw std::runtime_error("Invalid reshape");
    }
    dims = { width, height, channels };
}

template<typename T>
void Tensor<T>::resize(const std::vector<int> &dimensions)
{
    if (multiplyAllElements(dimensions) != elementCount())
    {
        delete[] data;
        data = new T[multiplyAllElements(dimensions)];
    }

    dims = dimensions;
    std::fill_n(data, elementCount(), T{0});
}

template<typename T>
void Tensor<T>::flatten()
{
    std::vector<int> newDims(1);
    newDims[0] = elementCount();
    reshape(newDims);
}

template<typename T>
void Tensor<T>::fillwith(T scalar)
{
    std::fill_n(data, elementCount(), scalar);
}

template<typename T>
void Tensor<T>::zero()
{
    memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
int Tensor<T>::elementCount() const
{
    return multiplyAllElements(dims);
}

template<typename T>
std::vector<int> Tensor<T>::dimensions() const
{
    return dims;
}

template<typename T>
int Tensor<T>::dimensionCount() const
{
    return dims.size();
}

template<typename T>
Tensor<T> Tensor<T>::pad(int horizontalPad, int verticalPad, T padValue) const
{
    assert(this->dimensionCount() == 3);

    int channelLen = channelLength();
    int rowLen = rowLength();
    int paddedRowLen = rowLen + 2 * horizontalPad;
    int columnLen = columnLength();
    int paddedColumnLen = columnLen + 2 * verticalPad;

    Tensor<T> padded(channelLen, paddedColumnLen, paddedRowLen);
    padded.fillwith(padValue);

    for (int channel = 0; channel < channelLen; channel++)
    {
        for (int i = 0; i < columnLen; i++)
        {
            vectorCopy(&padded(channel, verticalPad + i, horizontalPad), this->operator()(channel, i, 0), rowLen);
        }
    }

    return padded;
}

template<typename T>
Tensor<T> Tensor<T>::unpad(int horizontalPad, int verticalPad) const
{
    assert(this->dimensionCount() == 3);

    int channelLen = channelLength();
    int unpaddedRowLen = rowLength() - 2 * horizontalPad;
    int unpaddedColumnLen = columnLength() - 2 * verticalPad;

    Tensor<T> unpadded(channelLen, unpaddedColumnLen, unpaddedRowLen);

    for (int channel = 0; channel < channelLen; channel++)
    {
        for (int i = 0; i < unpaddedColumnLen; i++)
        {
            vectorCopy(&unpadded(channel,i,0), this->operator()(channel, verticalPad + i, horizontalPad), unpaddedRowLen);
        }
    }

    return unpadded;
}


template<typename T>
double Tensor<T>::mean()
{
    int els = elementCount();
    double result = 0;
    for (int i = 0; i < els ; i++)
    {
        result += (data[i] - result) / (i + 1);
    }

    return result;
}

template<typename T>
T Tensor<T>::max()
{
    int els = elementCount();
    T result = data[0];

    for (int i = 1; i < els; i++)
    {
        if (data[i] > result)
        {
            result = data[i];
        }
    }

    return result;
}

}
