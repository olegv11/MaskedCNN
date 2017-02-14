#pragma once
#include <vector>
#include <functional>
#include <numeric>
#include <cstring>

namespace MaskedCNN
{

// Better to make it explicit
class shallow_copy {};

template<typename T>
class Tensor
{
public:
    Tensor();
    Tensor(std::vector<int> &dimensions);
    Tensor(int width, int height, int channels);
    Tensor(const Tensor<T>& other); // deep copy
    Tensor(const Tensor<T>& other, shallow_copy) noexcept;
    Tensor(Tensor<T>&& other) noexcept;
    Tensor<T>& operator=(const Tensor<T>& other);
    Tensor<T>& operator=(Tensor<T>&& other) noexcept;
    ~Tensor();

    size_t width() const { return dims[0]; }
    size_t height() const { return dims[1]; }
    size_t channels() const { return dims[2]; }

    T& operator[](size_t index);
    const T& operator[](size_t index) const;

    T& operator()(int x, int y, int channel);
    const T& operator()(int x, int y, int channel) const;

    bool sameShape(const Tensor& other) const;

    void reshape(std::vector<int> &dimensions);
    void reshape(int width, int height, int channels);
    void flatten();

    int elementCount() const;
    std::vector<int> dimensions() const;
    int dimensionCount() const;

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
Tensor<T>::Tensor(int width, int height, int channels)
    :dims{width, height, channels}, isShallow(false)
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
T& Tensor<T>::operator()(int x, int y, int channel)
{
    assert(dimensionCount() == 3);
    assert(x < this->dims[0]);
    assert(y < this->dims[1]);
    assert(channel < this->dims[2]);
    return data[channel * dims[1] * dims[0] + y * dims[0] + x];
}

template<typename T>
const T& Tensor<T>::operator()(int x, int y, int channel) const
{
   assert(dimensionCount() == 3);
   assert(x < this->dims[0]);
   assert(y < this->dims[1]);
   assert(channel < this->dims[2]);
   return data[channel * dims[1] * dims[0] + y * dims[0] + x];
}


template<typename T>
bool Tensor<T>::sameShape(const Tensor &other) const
{
    return dims == other.dims;
}

template<typename T>
void Tensor<T>::reshape(std::vector<int> &dimensions)
{
    int newElementCount = std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<double>());
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
void Tensor<T>::flatten()
{
    std::vector<int> newDims(dimensions().size());
    newDims[0] = elementCount();
    for (int i = 1; i < newDims.size(); i++)
    {
        newDims = 1;
    }

    reshape(newDims);
}

template<typename T>
int Tensor<T>::elementCount() const
{
    return std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<double>());
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

}
