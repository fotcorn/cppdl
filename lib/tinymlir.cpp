#include "cppdl/tinymlir.h"

namespace tmlir {
// OpBuilder
Op OpBuilder::build() { return Op(); }

// TypeBuilder
Type TypeBuilder::build() { return Type(); }

// FunctionTypeBuilder
FunctionType FunctionTypeBuilder::build() { return FunctionType(); }

// BlockBuilder
Block BlockBuilder::build() { return Block(name); }

// Op
Type Op::result(size_t index) { return Type(); }
Region Op::region() { return Region(); }

// Module
TypeBuilder Module::type(std::string name) { return TypeBuilder(name); }
FunctionTypeBuilder Module::functionType() { return FunctionTypeBuilder(); }
OpBuilder Module::op(std::string operation) { return OpBuilder(operation); }

// Region
BlockBuilder Region::block(std::string name) { return BlockBuilder(name); }

// Block
OpBuilder Block::op(std::string operation) { return OpBuilder(operation); }
} // namespace tmlir
