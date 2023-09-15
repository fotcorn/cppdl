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
      // TODO: take offset and stride into account.
      result.data.get()[i] = func(data.get()[i]);
    }
    return result;
  }

  tensor<T> add(T op) const {
    return apply([op](T val) { return val + op; });
  }

  tensor<T> sub(T op) const {
    return apply([op](T val) { return val - op; });
  }

  tensor<T> mul(T op) const {
    return apply([op](T val) { return val * op; });
  }

  tensor<T> div(T op) const {
    if (op == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    return apply([op](T val) { return val / op; });
  }

  tensor<T> relu() const {
    return apply([](T val) { return std::max<T>(0, val); });
  }

  // Tensor ops.
  tensor<T> apply(const tensor<T> &op, std::function<T(T, T)> func) const {
    auto &op1 = *this;
    auto &op2 = op;

    // Shape is padded by 1's at the front.
    size_t length = std::max(op1.shape.size(), op2.shape.size());
    auto shapeOp1 = std::vector<size_t>(length, 1);
    auto shapeOp2 = std::vector<size_t>(length, 1);
    std::copy_backward(op1.shape.begin(), op1.shape.end(), shapeOp1.end());
    std::copy_backward(op2.shape.begin(), op2.shape.end(), shapeOp2.end());

    // Strides are padded by 1's at the back.
    auto strideOp1 = op1.strides;
    auto strideOp2 = op2.strides;
    ssize_t diff = strideOp1.size() - strideOp2.size();
    if (diff > 0) {
      for (ssize_t i = 0; i < diff; i++) {
        strideOp2.push_back(1);
      }
    } else if (diff < 0) {
      for (ssize_t i = diff; i < 0; i++) {
        strideOp1.push_back(1);
      }
    }

    assert(shapeOp1.size() == shapeOp2.size());
    assert(strideOp1.size() == strideOp2.size());

    for (size_t i = 0; i < shapeOp1.size(); i++) {
      if (!(shapeOp1[i] == shapeOp2[i] || shapeOp1[i] == 1 ||
            shapeOp2[i] == 1)) {
        throw std::runtime_error(
            "incompatible shapes for arithmetic operation");
      }
    }

    if (shapeOp1.size() == 1) {
      size_t dimOp1 = shapeOp1[0];
      size_t dimOp2 = shapeOp2[0];
      size_t maxDim = std::max(dimOp1, dimOp2);
      tensor<T> res = tensor<T>({maxDim});
      for (size_t i = 0; i < maxDim; i++) {
        res.data[i] = op1.data[op1.offset + strideOp1[0] * (i % dimOp1)] +
                      op2.data[op2.offset + strideOp2[0] * (i % dimOp2)];
      }
      return res;
    }

    if (shapeOp1.size() == 2) {
      size_t dim0Max = std::max(shapeOp1[0], shapeOp2[0]);
      size_t dim1Max = std::max(shapeOp1[1], shapeOp2[1]);

      tensor<T> res = tensor<T>({dim0Max, dim1Max});
      for (size_t dim0 = 0; dim0 < dim0Max; dim0++) {
        for (size_t dim1 = 0; dim1 < dim1Max; dim1++) {
          T index1 = op1.offset + (dim0 % shapeOp1[0]) * strideOp1[0] +
                     (dim1 % shapeOp1[1]) * strideOp1[1];
          T index2 = op2.offset + (dim0 % shapeOp2[0]) * strideOp2[0] +
                     (dim1 % shapeOp2[1]) * strideOp2[1];
          res.data[dim0 * dim0Max + dim1] =
              func(op1.data[index1], op2.data[index2]);
        }
      }
      return res;
    }

    throw std::runtime_error("unsupported shapes for arithmetic operation");
  }

  tensor<T> add(const tensor<T> &op) const {
    return apply(op, [](T v1, T v2) { return v1 + v2; });
  }

  tensor<T> sub(const tensor<T> &op) const {
    return apply(op, [](T v1, T v2) { return v1 - v2; });
  }

  tensor<T> mul(const tensor<T> &op) const {
    return apply(op, [](T v1, T v2) { return v1 * v2; });
  }

  tensor<T> div(const tensor<T> &op) const {
    return apply(op, [](T v1, T v2) {
      if (v2 == 0) {
        throw std::runtime_error("Division by zero is not allowed.");
      }
      return v1 / v2;
    });
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
