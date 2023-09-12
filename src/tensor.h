#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <span>
#include <sstream>
#include <vector>

// todo: implement elementwise with broadcast

template <typename T>
class tensor final {
public:
  tensor(std::vector<size_t> shape, T init = 0) : shape(shape) {
    offset = 0;
    size = 1;
    for (int dim : shape) {
      if (dim <= 0)
        throw std::runtime_error("Dimension size must be greater than 0");
      size *= dim;
    }
    data = std::shared_ptr<T[]>(new T[size]);
    std::fill_n(data.get(), size, init);

    strides.resize(shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  };

  T item() const {
    if (!(shape.size() == 1 && shape[0] == 1)) {
      throw std::runtime_error(
          "item() only works on tensors with one element.");
    }
    return data[offset];
  }

  static tensor<T> ones(const std::vector<size_t> &shape) {
    tensor<T> t(shape, 1);
    return t;
  }
  static tensor<T> vector(std::initializer_list<T> data) {
    std::vector<size_t> shape = {data.size()};
    tensor<T> t(shape);
    std::copy(data.begin(), data.end(), t.data.get());
    return t;
  }

  static tensor<T>
  matrix2d(std::initializer_list<std::initializer_list<T>> data) {
    if (data.size() == 0) {
      throw std::runtime_error("Input data cannot be empty.");
    }
    size_t subvector_size = data.begin()->size();
    for (const auto &subvector : data) {
      if (subvector.size() != subvector_size) {
        throw std::runtime_error("All subvectors must be the same size.");
      }
    }
    std::vector<size_t> shape = {data.size(), subvector_size};
    tensor<T> t(shape);
    T *ptr = t.data.get();
    for (const auto &subvector : data) {
      std::copy(subvector.begin(), subvector.end(), ptr);
      ptr += subvector_size;
    }
    return t;
  }

  tensor<T> operator[](const size_t index) {
    if (index >= shape[0]) {
      throw std::runtime_error("index out of range");
    }
    size_t new_offset = this->offset + this->strides[0] * index;
    if (shape.size() == 1) {
      std::vector<size_t> new_shape({1});
      std::vector<size_t> new_strides({1});
      return tensor<T>(this->data, new_offset, 1, new_shape, new_strides);
    }
    std::vector<size_t> new_shape(this->shape.begin() + 1, this->shape.end());
    std::vector<size_t> new_strides(this->strides.begin() + 1,
                                    this->strides.end());
    size_t new_size = 1;
    for (int dim : new_shape) {
      new_size *= dim;
    }
    return tensor<T>(this->data, new_offset, new_size, new_shape, new_strides);
  }

  // Elementwise ops.
  tensor<T> apply(std::function<T(T)> func) const {
    tensor<T> result(this->shape);
    for (size_t i = 0; i < result.size; i++) {
      result.data.get()[i] = func(data.get()[i]);
    }
    return result;
  }

  tensor<T> add(T op) const {
    return apply([&](T val) { return val + op; });
  }

  tensor<T> sub(T op) const {
    return apply([&](T val) { return val - op; });
  }

  tensor<T> mul(T op) const {
    return apply([&](T val) { return val * op; });
  }

  tensor<T> div(T op) const {
    if (op == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    return apply([&](T val) { return val / op; });
  }

  // Tensor ops.
  tensor<T> add(tensor<T> op) const {
    auto shape1 = shape;
    auto shape2 = op.shape;

    ssize_t diff = shape1.size() - shape2.size();
    if (diff > 0) {
      for (ssize_t i = 0; i < diff; i++) {
        shape2.push_back(1);
      }
    } else if (diff < 0) {
      for (ssize_t i = diff; i < 0; i++) {
        shape1.push_back(1);
      }
    }
    assert(shape1.size() == shape2.size());

    for (size_t i = 0; i < shape1.size(); i++) {
      if (!(shape1[i] == shape2[i] || shape1[i] == 1 || shape2[i] == 1)) {
        throw std::runtime_error(
            "incompatible shapes for arithmetic operation");
      }
    }

    if (shape1.size() == 1) {
      size_t dim1 = shape1[0];
      size_t dim2 = shape2[0];
      size_t dim = std::max(dim1, dim2);
      tensor<T> res = tensor<T>({dim});
      for (size_t i = 0; i < dim; i++) {
        res.data[i] = data[i % dim1] + op.data[i % dim2];
      }
      return res;
    }
    throw std::runtime_error("unsupported shapes for arithmetic operation");
  }

  std::string to_string() const {
    if (shape.size() > 2) {
      throw std::runtime_error(
          "to_string() only works on tensors with one or two dimensions.");
    }
    std::stringstream ss;
    if (shape.size() == 1) {
      ss << "[";
      for (size_t i = 0; i < shape[0]; i++) {
        ss << data[offset + i];
        if (i != shape[0] - 1) {
          ss << ", ";
        }
      }
      ss << "]";
    } else if (shape.size() == 2) {
      ss << "[\n";
      for (size_t i = 0; i < shape[0]; i++) {
        ss << "  [";
        for (size_t j = 0; j < shape[1]; j++) {
          ss << data[offset + i * strides[0] + j * strides[1]];
          if (j != shape[1] - 1) {
            ss << ", ";
          }
        }
        ss << "]\n";
      }
      ss << "]";
    }
    return ss.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const tensor<T> &t) {
    os << t.to_string();
    return os;
  }

  const std::vector<size_t> &getShape() const { return shape; }

private:
  tensor(std::shared_ptr<T[]> data, size_t offset, size_t size,
         std::vector<size_t> shape, std::vector<size_t> strides)
      : data(data), offset(offset), size(size), shape(shape), strides(strides) {
  }

  std::shared_ptr<T[]> data;
  size_t offset;
  size_t size;
  std::vector<size_t> shape;
  std::vector<size_t> strides;
};
