
#include <cmath>
#include <random>

#include <fmt/format.h>
#include <fmt/ranges.h>

class Random {
private:
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<> dis;

public:
  Random(unsigned int seed) : gen(seed), dis(0, 1.0) {}

  float getFloatZeroOne() { return dis(gen); }
};

int main() {
  Random r(42);

  constexpr float LEARNING_RATE = 0.01;
  constexpr int NUM_SAMPLES = 4;

  float w1 = r.getFloatZeroOne();
  float w2 = r.getFloatZeroOne();
  float b = r.getFloatZeroOne();

  float input[NUM_SAMPLES][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  float groundTruth[NUM_SAMPLES] = {0, 1, 1, 1}; // or
  // float groundTruth[NUM_SAMPLES] = {1, 0, 0, 0}; // nor
  // float groundTruth[NUM_SAMPLES] = {0, 0, 0, 1}; // and
  // float groundTruth[NUM_SAMPLES] = {1, 1, 1, 0}; // nand

  for (int epoch = 0; epoch < 100; epoch++) {
    int correct = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // Perceptron with bias
      float a1 = input[i][0] * w1;
      float a2 = input[i][1] * w2;
      float result = a1 + a2 + b;

      if ((result > 0.5 && groundTruth[i] > 0.5) ||
          (result < 0.5 && groundTruth[i] < 0.5)) {
        correct++;
      }

      float loss = groundTruth[i] - result;

      b += loss * LEARNING_RATE;
      w1 += (input[i][0] == 0 ? -1 : 1) * loss * LEARNING_RATE;
      w2 += (input[i][1] == 0 ? -1 : 1) * loss * LEARNING_RATE;
    }
    fmt::println("{}", correct);
  }

  fmt::println("w1: {}, w2: {}, b: {}", w1, w2, b);
}
