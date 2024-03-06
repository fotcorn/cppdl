#include "cppdl/serialization.h"
#include <gtest/gtest.h>
#include <sstream>

TEST(SerializationTest, SerializeDeserializeIntTensor) {
  auto tensor = Tensor<int>::vector({1, 2, 3, 4});
  std::stringstream ss;
  serializeTensor(tensor, ss);

  auto deserializedTensorOpt = deserializeTensor<int>(ss);
  ASSERT_TRUE(deserializedTensorOpt.has_value());
  auto deserializedTensor = deserializedTensorOpt.value();

  ASSERT_EQ(deserializedTensor.shape.size(), 1);

  ASSERT_EQ(deserializedTensor.data.get()[0], 1);
  ASSERT_EQ(deserializedTensor.data.get()[1], 2);
  ASSERT_EQ(deserializedTensor.data.get()[2], 3);
  ASSERT_EQ(deserializedTensor.data.get()[3], 4);
}

TEST(SerializationTest, SerializeDeserializeFloatTensor) {
  auto tensor = Tensor<float>::vector({1.1f, 2.2f, 3.3f});
  std::stringstream ss;
  serializeTensor(tensor, ss);

  auto deserializedTensorOpt = deserializeTensor<float>(ss);
  ASSERT_TRUE(deserializedTensorOpt.has_value());
  auto deserializedTensor = deserializedTensorOpt.value();

  ASSERT_EQ(deserializedTensor.shape.size(), 1);
  ASSERT_EQ(deserializedTensor.shape[0], 3);

  ASSERT_FLOAT_EQ(deserializedTensor.data.get()[0], 1.1f);
  ASSERT_FLOAT_EQ(deserializedTensor.data.get()[1], 2.2f);
  ASSERT_FLOAT_EQ(deserializedTensor.data.get()[2], 3.3f);
}

TEST(SerializationTest, SerializeDeserializeIntMatrix) {
  auto matrix = Tensor<int>::matrix2d({{1, 2}, {3, 4}});
  std::stringstream ss;
  serializeTensor(matrix, ss);

  auto deserializedMatrixOpt = deserializeTensor<int>(ss);
  ASSERT_TRUE(deserializedMatrixOpt.has_value());
  auto deserializedMatrix = deserializedMatrixOpt.value();

  ASSERT_EQ(deserializedMatrix.shape.size(), 2);
  ASSERT_EQ(deserializedMatrix.shape[0], 2);
  ASSERT_EQ(deserializedMatrix.shape[1], 2);

  ASSERT_EQ(deserializedMatrix.data.get()[0], 1);
  ASSERT_EQ(deserializedMatrix.data.get()[1], 2);
  ASSERT_EQ(deserializedMatrix.data.get()[2], 3);
  ASSERT_EQ(deserializedMatrix.data.get()[3], 4);
}

TEST(SerializationTest, SerializeDeserializeFloatMatrix) {
  auto matrix = Tensor<float>::matrix2d({{1.1f, 2.2f}, {3.3f, 4.4f}});
  std::stringstream ss;
  serializeTensor(matrix, ss);

  auto deserializedMatrixOpt = deserializeTensor<float>(ss);
  ASSERT_TRUE(deserializedMatrixOpt.has_value());
  auto deserializedMatrix = deserializedMatrixOpt.value();

  ASSERT_EQ(deserializedMatrix.shape.size(), 2);
  ASSERT_EQ(deserializedMatrix.shape[0], 2);
  ASSERT_EQ(deserializedMatrix.shape[1], 2);

  ASSERT_FLOAT_EQ(deserializedMatrix.data.get()[0], 1.1f);
  ASSERT_FLOAT_EQ(deserializedMatrix.data.get()[1], 2.2f);
  ASSERT_FLOAT_EQ(deserializedMatrix.data.get()[2], 3.3f);
  ASSERT_FLOAT_EQ(deserializedMatrix.data.get()[3], 4.4f);
}

TEST(SerializationTest, DeserializeValidFileHeader) {
  std::stringstream ss;
  writeFileHeader(ss);

  bool validHeader = readFileHeader(ss);
  ASSERT_TRUE(validHeader);
}

TEST(SerializationTest, DeserializeInvalidMagicNumber) {
  std::stringstream ss;
  ss.write("INVALID", 7);
  char validVersion = Version;
  ss.write(&validVersion, sizeof(validVersion));

  bool header = readFileHeader(ss);
  ASSERT_FALSE(header);
}

TEST(SerializationTest, DeserializeInvalidVersion) {
  std::stringstream ss;
  ss.write(MagicNumber, sizeof(MagicNumber));
  char invalidVersion = Version + 1;
  ss.write(&invalidVersion, sizeof(invalidVersion));

  bool header = readFileHeader(ss);
  ASSERT_FALSE(header);
}