#pragma once
#include <vector>
#include <functional>
#include <numeric>
#include <cstring>
#include <string>
#include <cmath>
#include "Util.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
namespace MaskedCNN
{

// Better to make it explicit
class shallow_copy {};

enum class DataPosition {
    UNDEFINED,
    CPU,
    GPU
};

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

    Tensor<T>& toGpu();
    Tensor<T>& toCpu();

    DataPosition position() const;

    bool operator==(const Tensor<T>& other) const;

    int rowLength() const { return dims[dimensionCount() - 1]; }
    int columnLength() const { return dims[dimensionCount() - 2]; }
    int channelLength() const { return dims[dimensionCount() - 3]; }
    int channel2Length() const { return dims[dimensionCount() - 4]; }

    T* dataAddress();
    const T* dataAddress() const;

    T* gpuDataAddress();
    const T* gpuDataAddress() const;

    inline T& operator[](size_t index);
    inline const T& operator[](size_t index) const;


    inline T& operator()(int row, int column);
    inline const T& operator()(int row, int column) const;

    inline T& operator()(int channel, int row, int column);
    inline const T& operator()(int channel, int row, int column) const;

    inline T& operator()(int channel2, int channel, int row, int column);
    inline const T& operator()(int channel2, int channel, int row, int column) const;

    bool sameShape(const Tensor& other) const;

    void reshape(const std::vector<int> &dimensions) const;
    void reshape(int rowLength, int columnLength, int channelLength);
    void resize(const std::vector<int> &dimensions);
    void flatten();
    void fillwith(T scalar);

    void zero();

    int elementCount() const;
    int nonZeroCount() const;
    std::vector<int> dimensions() const;
    int dimensionCount() const;

    double mean();
    T max();
    void add(T value);
    void add(float b, float g, float r);
    void mul(T value);

    double howFilled();
    std::string toString() const;

private:
    mutable std::vector<int> dims;
    T *data = nullptr;
    T *gpuData = nullptr;
    bool isShallow;
    DataPosition dataPosition = DataPosition::UNDEFINED;
};

template<typename T>
Tensor<T>::Tensor()
    :data(nullptr), isShallow(false), dataPosition(DataPosition::UNDEFINED)
{
}

template<typename T>
Tensor<T>::Tensor(std::vector<int> &dimensions)
    :dims(dimensions), isShallow(false), dataPosition(DataPosition::CPU)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(std::vector<int> &&dimensions)
    :dims(std::move(dimensions)), isShallow(false), dataPosition(DataPosition::CPU)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(int channelLength, int columnLength, int rowLength)
    :dims{channelLength, columnLength, rowLength}, isShallow(false), dataPosition(DataPosition::CPU)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(int channelLength2, int channelLength, int columnLength, int rowLength)
    :dims{channelLength2, channelLength, columnLength, rowLength}, isShallow(false), dataPosition(DataPosition::CPU)
{
    data = new T[elementCount()];
    std::memset(data, 0, elementCount() * sizeof(T));
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other)
    :dims(other.dims), isShallow(false), dataPosition(other.dataPosition)
{
    switch (other.dataPosition)
    {
    case DataPosition::UNDEFINED:
        throw std::runtime_error("POSITION UNDEFINED");
        break;
    case DataPosition::CPU:
        data = new T[other.elementCount()];
        std::memcpy(data, other.data, other.elementCount() * sizeof(T));
        break;
    case DataPosition::GPU:
        cudaMalloc(&gpuData, other.elementCount());
        cudaMemcpy(gpuData, other.gpuData, other.elementCount() * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other, shallow_copy) noexcept
    :dims(other.dims), data(other.data), gpuData(other.gpuData), isShallow(true), dataPosition(other.dataPosition)
{
}

template<typename T>
Tensor<T>::Tensor(Tensor<T> &&other) noexcept
    :dims(std::move(other.dims)), data(std::move(other.data)), gpuData(std::move(other.gpuData)), dataPosition(other.dataPosition)
{
    isShallow = other.isShallow;
    other.data = nullptr;
    other.gpuData = nullptr;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other)
{
    if (this == &other)
    {
        return *this;
    }

    switch (other.dataPosition)
    {
    case DataPosition::UNDEFINED:
        throw std::runtime_error("POSITION UNDEFINED");
        break;
    case DataPosition::CPU:
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
        break;
    case DataPosition::GPU:
        if (elementCount() == other.elementCount())
        {
            dims = other.dims;
            cudaMemcpy(gpuData, other.gpuData, elementCount() * sizeof(T), cudaMemcpyDeviceToDevice);
        }
        else
        {
            cudaFree(gpuData);
            dims = other.dims;
            cudaMalloc(&gpuData, elementCount());
        }
        break;
    }

    isShallow = false;
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept
{
    dims = std::move(other.dims);
    data = std::move(other.data);
    gpuData = std::move(other.data);
    dataPosition = other.dataPosition;

    isShallow = other.isShallow;
    other.data = nullptr;
    other.gpuData = nullptr;
    other.dataPosition = DataPosition::UNDEFINED;

    return *this;
}

template<typename T>
Tensor<T>::~Tensor()
{
    if (!isShallow)
    {
        switch (dataPosition)
        {
        case DataPosition::UNDEFINED:
            break;
        case DataPosition::CPU:
            delete[] data;
            data = nullptr;
            break;
        case DataPosition::GPU:
            cudaFree(gpuData);
            gpuData = nullptr;
            break;
        }
    }
    dataPosition = DataPosition::UNDEFINED;
}

template<typename T>
Tensor<T>& Tensor<T>::toGpu()
{
    switch (dataPosition)
    {
    case DataPosition::UNDEFINED:
        gpuData = nullptr;
        dataPosition = DataPosition::GPU;
        break;
    case DataPosition::CPU:
        cudaMalloc(&gpuData, elementCount() * sizeof(T));
        cudaMemcpy(gpuData, data, elementCount() * sizeof(T), cudaMemcpyHostToDevice);
        delete[] data;
        data = nullptr;
        dataPosition = DataPosition::GPU;
        break;
    case DataPosition::GPU:
        break;
    }
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::toCpu()
{
    switch (dataPosition)
    {
    case DataPosition::UNDEFINED:
        data = nullptr;
        dataPosition = DataPosition::CPU;
        break;
    case DataPosition::GPU:
        data = new T[elementCount()];
        cudaMemcpy(data, gpuData, elementCount() * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(gpuData);
        gpuData = nullptr;
        dataPosition = DataPosition::CPU;
        break;
    case DataPosition::CPU:
        break;
    }
    return *this;
}

template<typename T>
DataPosition Tensor<T>::position() const
{
    return dataPosition;
}

template<typename T>
T* Tensor<T>::gpuDataAddress()
{
    assert(dataPosition == DataPosition::GPU);
    return gpuData;
}

template<typename T>
const T* Tensor<T>::gpuDataAddress() const
{
    assert(dataPosition == DataPosition::GPU);
    return gpuData;
}

template<typename T>
bool Tensor<T>::operator==(const Tensor<T> &other) const
{
    if (&other == this) return true;
    if (dims != other.dims) return false;

    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    int els = elementCount();
    for (int i = 0; i < els; i++)
    {
        if (data[i] != other.data[i])
        {
            return false;
        }
    }

    return true;
}

template<typename T>
T* Tensor<T>::dataAddress()
{
    assert(dataPosition == DataPosition::CPU);
    return data;
}

template<typename T>
const T* Tensor<T>::dataAddress() const
{
    assert(dataPosition == DataPosition::CPU);
    return data;
}

template<typename T>
T& Tensor<T>::operator[](size_t index)
{
    assert(dataPosition == DataPosition::CPU);
    return data[index];
}

template<typename T>
const T& Tensor<T>::operator[](size_t index) const
{
    assert(dataPosition == DataPosition::CPU);
    return data[index];
}

template<typename T>
inline T& Tensor<T>::operator()(int row, int column)
{
    assert(dataPosition == DataPosition::CPU);
    assert(dimensionCount() == 2);
    assert(column < this->dims[1] && column >= 0);
    assert(row < this->dims[0] && row >= 0);
    return data[row * dims[1] + column];
}

template<typename T>
inline const T& Tensor<T>::operator()(int row, int column) const
{
    assert(dataPosition == DataPosition::CPU);
    assert(dimensionCount() == 2);
    assert(column < this->dims[1] && column >= 0);
    assert(row < this->dims[0] && row >= 0);
    return data[row * dims[1] + column];
}

template<typename T>
inline T& Tensor<T>::operator()(int channel, int row, int column)
{
    assert(dataPosition == DataPosition::CPU);
    assert(dimensionCount() == 3);
    assert(column < this->dims[2] && column >= 0);
    assert(row < this->dims[1] && row >= 0);
    assert(channel < this->dims[0] && channel >= 0);
    return data[(channel * dims[1] + row) * dims[2] + column];
}

template<typename T>
inline const T& Tensor<T>::operator()(int channel, int row, int column) const
{
   assert(dataPosition == DataPosition::CPU);
   assert(dimensionCount() == 3);
   assert(column < this->dims[2] && column >= 0);
   assert(row < this->dims[1] && row >= 0);
   assert(channel < this->dims[0] && channel >= 0);

   return data[(channel * dims[1] + row) * dims[2] + column];
}

template<typename T>
inline T& Tensor<T>::operator()(int channel2, int channel, int row, int column)
{
    assert(dataPosition == DataPosition::CPU);
    assert(dimensionCount() == 4);
    assert(column < this->dims[3] && column >= 0);
    assert(row < this->dims[2] && row >= 0);
    assert(channel < this->dims[1] && channel >= 0);
    assert(channel2 < this->dims[0] && channel2 >= 0);
    return data[((channel2 * dims[1] + channel) * dims[2] + row) * dims[3] + column];
}

template<typename T>
inline const T& Tensor<T>::operator()(int channel2, int channel, int row, int column) const
{
    assert(dataPosition == DataPosition::CPU);
    assert(dimensionCount() == 4);
    assert(column < this->dims[3] && column >= 0);
    assert(row < this->dims[2] && row >= 0);
    assert(channel < this->dims[1] && channel >= 0);
    assert(channel2 < this->dims[0] && channel2 >= 0);
    return data[((channel2 * dims[1] + channel) * dims[2] + row) * dims[3] + column];
}


template<typename T>
inline bool Tensor<T>::sameShape(const Tensor &other) const
{
    return dims == other.dims;
}

template<typename T>
void Tensor<T>::reshape(const std::vector<int> &dimensions) const
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
        switch(dataPosition)
        {
        case DataPosition::UNDEFINED:
            data = new T[multiplyAllElements(dimensions)];
            std::fill_n(data, elementCount(), T{0});
            dataPosition = DataPosition::CPU;
            break;
        case DataPosition::CPU:
            delete[] data;
            data = new T[multiplyAllElements(dimensions)];
            std::fill_n(data, elementCount(), T{0});
            break;
        case DataPosition::GPU:
            cudaFree(gpuData);
            cudaMalloc(&gpuData, multiplyAllElements(dimensions) * sizeof(T));
            cudaMemset(gpuData, 0, multiplyAllElements(dimensions) * sizeof(T));
            break;
        }
    }

    dims = dimensions;
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
    switch(dataPosition)
    {
    case DataPosition::UNDEFINED:
        throw std::runtime_error("UNDEFINED POSITION");
        break;
    case DataPosition::CPU:
        std::fill_n(data, elementCount(), scalar);
        break;
    case DataPosition::GPU:
        toCpu();
        std::fill_n(data, elementCount(), scalar);
        toGpu();
        break;
    }
}

template<typename T>
void Tensor<T>::zero()
{
    switch(dataPosition)
    {
    case DataPosition::UNDEFINED:
        throw std::runtime_error("UNDEFINED POSITION");
        break;
    case DataPosition::CPU:
        memset(data, 0, elementCount() * sizeof(T));
        break;
    case DataPosition::GPU:
        cudaMemset(gpuData, 0, elementCount() * sizeof(T));
        break;
    }

}

template<typename T>
int Tensor<T>::elementCount() const
{
    if (dims.size() == 0) return 0;
    return multiplyAllElements(dims);
}

template<typename T>
int Tensor<T>::nonZeroCount() const
{
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    int count = 0;
    int els = elementCount();
    for (int i = 0; i < els; i++)
    {
        if (data[i] != 0)
        {
            count++;
        }
    }

    return count;
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
double Tensor<T>::mean()
{
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

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
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    int els = elementCount();
    T result = std::abs(data[0]);

    for (int i = 1; i < els; i++)
    {
        if (std::abs(data[i]) > result)
        {
            result = std::abs(data[i]);
        }
    }

    return result;
}

template<typename T>
void Tensor<T>::add(T value)
{
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    int els = elementCount();
    for (int i = 0; i < els; i++)
    {
        data[i] += value;
    }
}

template<typename T>
void Tensor<T>::add(float b, float g, float r)
{
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    for (int y = 0; y < columnLength(); y++)
    {
        for (int x = 0; x < rowLength(); x++)
        {
            operator()(0,y,x) += b;
            operator()(1,y,x) += g;
            operator()(2,y,x) += r;
        }
    }
}

template<typename T>
void Tensor<T>::mul(T value)
{
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    int els = elementCount();
    for (int i = 0; i < els; i++)
    {
        data[i] *= value;
    }
}

template<typename T>
double Tensor<T>::howFilled()
{
    return (double)nonZeroCount() / (double)elementCount();
}

template<typename T>
std::string Tensor<T>::toString() const
{
    if (dataPosition != DataPosition::CPU)
    {
        throw std::runtime_error("AVAILABLE ONLY ON CPU");
    }

    assert(dimensionCount() <= 2);
    std::string result;

    for (int i = 0; i < dims[0]; i++)
    {
        for (int j = 0; j < dims[1]; j++)
        {
            result += std::to_string(data[j + i * dims[1]]);
            result += " ";
        }
        result += "\n";
    }

    return result;
}


template <typename T>
inline void normalize(std::vector<Tensor<T>>& examples)
{
    double mean = 0;
    for (size_t example = 0; example < examples.size(); example++)
    {
        mean += examples[example].mean();
    }

    mean /= examples.size();

    for (size_t example = 0; example < examples.size(); example++)
    {
        examples[example].add(-mean);
    }


    double max = examples[0].max();
    for (size_t example = 1; example < examples.size(); example++)
    {
        double t = examples[example].max();
        if (t > max)
        {
            max = t;
        }
    }

    for (size_t example = 0; example < examples.size(); example++)
    {
        examples[example].mul(1.0 / max);
    }

}

}
