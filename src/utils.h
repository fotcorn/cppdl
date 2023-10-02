#include <vector>

template <typename T>
void printVector(const std::vector<T> &vec) {
  std::cout << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    std::cout << vec[i];
    if (i != vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
