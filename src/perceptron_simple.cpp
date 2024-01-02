
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

  constexpr float LEARNING_RATE = 0.1;
  constexpr int NUM_SAMPLES = 4;

  float w1 = r.getFloatZeroOne();
  float w2 = r.getFloatZeroOne();

  float input[NUM_SAMPLES][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  float groundTruth[NUM_SAMPLES] = {0, 1, 1, 1}; // or
  // float groundTruth[NUM_SAMPLES] = {1, 0, 0, 0}; // nor
  // float groundTruth[NUM_SAMPLES] = {0, 0, 0, 1}; // and

  float activations[4][2];
  float loss[4];

  float totalLoss = 0;

  for (int epoch = 0; epoch < 100; epoch++) {
    int correct = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
      activations[i][0] = input[i][0] * w1;
      activations[i][1] = input[i][1] * w2;
      float result = activations[i][0] + activations[i][1];

      loss[i] = result - groundTruth[i];

      if ((result > 0.45 && groundTruth[i] > 0.5) ||
          (result < 0.3 && groundTruth[i] < 0.5)) {
        correct++;
      }

      totalLoss += loss[i] * loss[i];
    }

    totalLoss /= NUM_SAMPLES;

    float w1Grad = 0;
    float w2Grad = 0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
      float lossGrad = loss[i] * totalLoss * 2;
      w1Grad += (input[i][0] == 0 ? -1 : 1) * lossGrad;
      w2Grad += (input[i][1] == 0 ? -1 : 1) * lossGrad;

      /*
      a1 = i1 * w1;
      a2 = i2 * w2;
      a3 = a1 + a2;
      a4 = a3 - gt;
      a5 = a4 * a4;

      a5Grad = totalLoss;
      a4Grad = a4 * totalLoss; // Double loss?
      a3Grad = a4Grad;
      a1Grad = a3Grad;
      a2Grad = a3Grad;
      w1Grad = i1 * a1Grad;
      w2Grad = i2 * a2Grad;
      */
    }

    w1 -= w1Grad * LEARNING_RATE;
    w2 -= w2Grad * LEARNING_RATE;

    fmt::println(
        "loss: {}, correct: {}, w1: {}, w1grad: {}, w2: {}, w2grad: {}",
        totalLoss, correct, w1, w1Grad, w2, w2Grad);
  }
}
