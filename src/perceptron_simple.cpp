
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

constexpr float LEARNING_RATE = 0.01;
constexpr int NUM_SAMPLES = 4;

struct Dataset {
  const char *name;
  float groundTruth[NUM_SAMPLES];
};

void learn(float groundTruth[NUM_SAMPLES]);

int main() {

  Dataset datasets[] = {{"or", {0, 1, 1, 1}},  {"nor", {1, 0, 0, 0}},
                        {"and", {0, 0, 0, 1}}, {"nand", {1, 1, 1, 0}},
                        {"xor", {0, 1, 1, 0}}, {"nxor", {1, 0, 0, 1}}};

  for (Dataset &dataset : datasets) {
    fmt::println("Dataset: {}", dataset.name);
    learn(dataset.groundTruth);
  }
}

void learn(float groundTruth[NUM_SAMPLES]) {
  Random r(42);

  float l1p1w1 = r.getFloatZeroOne();
  float l1p1w2 = r.getFloatZeroOne();
  float l1p1b = r.getFloatZeroOne();

  float l1p2w1 = r.getFloatZeroOne();
  float l1p2w2 = r.getFloatZeroOne();
  float l1p2b = r.getFloatZeroOne();

  float l2w1 = r.getFloatZeroOne();
  float l2w2 = r.getFloatZeroOne();
  float l2b = r.getFloatZeroOne();

  float input[NUM_SAMPLES][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

  int correct, epoch;
  for (epoch = 0; epoch < 100000; epoch++) {
    correct = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // Perceptron with bias
      float a1 = input[i][0] * l1p1w1;
      float a2 = input[i][1] * l1p1w2;
      float p1Res = a1 + a2 + l1p1b;

      float a3 = input[i][0] * l1p2w1;
      float a4 = input[i][1] * l1p2w2;
      float p2Res = a3 + a4 + l1p2b;

      float result = p1Res * l2w1 + p2Res * l2w2 + l2b;

      if ((result > 0 && groundTruth[i] == 1) ||
          (result <= 0 && groundTruth[i] == 0)) {
        correct++;
      }

      float loss = groundTruth[i] - result;

      // Back propagation
      float l2w1Grad = p1Res * loss;
      float l2w2Grad = p2Res * loss;

      // Gradient descent
      l2b += loss * LEARNING_RATE;
      l2w1 += l2w1Grad * LEARNING_RATE;
      l2w2 += l2w2Grad * LEARNING_RATE;

      l1p1b += (l2w1Grad + l2w2Grad) * LEARNING_RATE;
      l1p2b += (l2w1Grad + l2w2Grad) * LEARNING_RATE;

      l1p1w1 += input[i][0] * (l2w1Grad + l2w2Grad) * LEARNING_RATE;
      l1p1w2 += input[i][1] * (l2w1Grad + l2w2Grad) * LEARNING_RATE;

      l1p2w1 += input[i][0] * (l2w1Grad + l2w2Grad) * LEARNING_RATE;
      l1p2w2 += input[i][1] * (l2w1Grad + l2w2Grad) * LEARNING_RATE;
    }
    if (correct == NUM_SAMPLES) {
      fmt::println("Solution found after {} epochs", epoch);
      return;
    }
  }

  fmt::println("No solution found after {} epochs", epoch);
}
