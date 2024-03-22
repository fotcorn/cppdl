#include <vector>

class TraceTensor {};

class TraceContext {
public:
  TraceTensor zeroes(std::vector<std::size_t> shape) { return TraceTensor(); }
  TraceTensor ones(std::vector<std::size_t> shape) { return TraceTensor(); }
  TraceTensor random(std::vector<std::size_t> shape) { return TraceTensor(); }
  TraceTensor vector(std::vector<float> values) { return TraceTensor(); }
  TraceTensor matrix2d(std::vector<std::vector<float>> values) {
    return TraceTensor();
  }
};

enum OpType {
  Add,
  Mul,
};

struct Operation {
  OpType type;

  Operation *in1;
  Operation *in2;

  tensor<float> forward;
  tensor<float> grad;
};

template <typename T>
struct ReLU final : public ElementwiseOp {
  T forward(T input) final { return std::max(0, input); }
  T backward(T input) { return input <= 0 ? 0 : 1; }
};

template <typename T>
struct Add final : public ElementwiseParamOp {
  T forward(T input, T param) final { return input + param; }
  T backward(T input, T outGrad) { return outGrad; }
};

template <typename T>
struct Mul final : public ElementwiseParamOp {
  T forward(T input, T param) final { return input * param; }
  T backward(T input, T outGrad) { return input * outGrad; }
};

template <typename T>
struct MatMul final : public TensorParamOp {
  tensor<T> forward(tensor<T> input, tensor<T> param) final { return 1.0; }
  tensor<T> backward(tensor<T> param, tensor<T> outGrad) {
    return param * outGrad;
  }
};

template <typename T>
class Net {
  MatMul<T> w1;
  Add<T> b1;

  Net() {
    w1 = register<MatMul<T>>("w1");
    b1 = register<Add<T>>("b1");
  }

  tensor<T> forward(tensor<T> input) {
    auto resLayer1 = F::relu(b1(w1(input)));
    auto resLayer2 = F::relu(b2(w2(resLayer1)));
    return b3(w3(resLayer2));
  }
};

template <typename DataTensor, typename ParamTensor>
DataTensor net(DataTensor input, ParamTensor w1, ParamTensor b1, ParamTensor w2,
               ParamTensor b2, ParamTensor w3, ParamTensor b3) {
  auto o1 = (input.matmul(w1.transpose()) + b1).relu();
  auto o2 = (o1.matmul(w2.transpose()) + b2).relu();
  return o2.matmul(w3) + b3;
}

template <typename T>
struct trace_tensor {};

template <typename T>
struct placeholder_tensor {
  placeholder_tensor(std::initializer_list<T> shape) {}
};

int main() {
  auto w1 = tensor<float>::random(LAYER_0_WEIGHTS.shape);
  auto b1 = tensor<float>::random(LAYER_0_BIASES.shape);
  auto w2 = tensor<float>::random(LAYER_1_WEIGHTS.shape);
  auto b2 = tensor<float>::random(LAYER_1_BIASES.shape);
  auto w3 = tensor<float>::random(LAYER_2_WEIGHTS.shape);
  auto b3 = tensor<float>::random(LAYER_2_BIASES.shape);

  trace_tensor<float> input;

  placeholder_tensor<float> w1({1, 2});
  placeholder_tensor<float> b1({1, 2});
  placeholder_tensor<float> w2({1, 2});
  placeholder_tensor<float> b2({1, 2});
  placeholder_tensor<float> w3({1, 2});
  placeholder_tensor<float> b3({1, 2});

  trace_tensor<float> forwardNet = net(input, w1, b1, w2, b2, w3, b3);
  return 0;
}