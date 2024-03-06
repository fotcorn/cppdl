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
