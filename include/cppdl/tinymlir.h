#pragma once

#include <string>

namespace tmlir {

class Type {};
class TypeBuilder {
  TypeBuilder(std::string name) {}

public:
  TypeBuilder &param(std::string) { return *this; }
  TypeBuilder &param(const Type &type) { return *this; }
  Type build(){};

  friend class Module;
};

class FunctionType {};
class FunctionTypeBuilder {
  FunctionTypeBuilder() {}

public:
  FunctionTypeBuilder &param(const Type &type) { return *this; }
  FunctionTypeBuilder &returnType(const Type &type) { return *this; }
  FunctionType build() { return FunctionType(); };

  friend class Module;
};

class Op;
class OpBuilder {
  OpBuilder(std::string operation) {}

public:
  OpBuilder &param(const Type &type) { return *this; }
  OpBuilder &type(const Type &type) { return *this; }
  OpBuilder &attr(const std::string &name, const std::string &value) {
    return *this;
  }
  OpBuilder &attr(const std::string &name, const Type &type) { return *this; }
  OpBuilder &attr(const std::string &name, const FunctionType &type) {
    return *this;
  }
  Op build();

  friend class Module;
  friend class Block;
};

class Block {
  Block(std::string name) {}

public:
  OpBuilder op(std::string operation) { return OpBuilder(operation); }
  friend class BlockBuilder;
};
class BlockBuilder {
  std::string name;

public:
  BlockBuilder(std::string name) : name(name) {}
  Block build() { return Block(name); }
  BlockBuilder &param(const std::string &name, const Type &type) {
    return *this;
  }
};

class Region {
public:
  BlockBuilder block(std::string name) { return BlockBuilder(name); }
};

class Op {
public:
  Type result(size_t index) { return Type(); }
  Region region() { return Region(); }
};

class Module {
public:
  TypeBuilder type(std::string name) { return TypeBuilder(name); }
  FunctionTypeBuilder functionType() { return FunctionTypeBuilder(); }
  OpBuilder op(std::string operation) { return OpBuilder(operation); }
};

}; // namespace tmlir
