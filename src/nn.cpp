#include "data/nn.h"
#include "data/dataset.h"

#include "nn.h"

int main() {
  /*
  LinearLayer layer0(LAYER_0_WEIGHTS.transpose(), LAYER_0_BIASES);
  ReLU layer0Activation;
  LinearLayer layer1(LAYER_1_WEIGHTS.transpose(), LAYER_1_BIASES);
  ReLU layer1Activation;
  LinearLayer layer2(LAYER_2_WEIGHTS, LAYER_2_BIASES);
  */

  LinearLayer layer0(2, 16);
  ReLU layer0Activation;
  LinearLayer layer1(16, 16);
  ReLU layer1Activation;
  LinearLayer layer2(16, 1);

  for (int epoch = 0; epoch < 1000; epoch++) {
    auto r1 = layer0.forward(datasetValues);
    auto r2 = layer0Activation.forward(r1);
    auto r3 = layer1.forward(r2);
    auto r4 = layer1Activation.forward(r3);
    auto result = layer2.forward(r4);

    int correct = 0;
    for (size_t i = 0; i < datasetValues.getShape()[0]; i++) {
      if (std::signbit(result[i].item()) ==
          std::signbit(datasetLabels[i].item())) {
        correct++;
      }
    }

    auto flatResult = result.reshape({100});

    float accuracy = static_cast<float>(correct) / datasetValues.getShape()[0];
    fmt::println("Accuracy: {}", accuracy);
  }

  return 0;
}
