// Adapted from https://github.com/iree-org/iree-template-compiler-cmake/ and
// https://github.com/openxla/iree/blob/main/tools/iree-run-mlir-main.cc
// https://iree.dev/reference/bindings/c-api/#usage_1
// https://github.com/openxla/iree/tree/main/runtime/src/iree/runtime/demo
//
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cinttypes>
#include <optional>
#include <vector>

#include <iree/compiler/embedding_api.h>

#include <fmt/core.h>

#include "cppdl/iree.h"

#define IREE_COMPILER_EXPECTED_API_MAJOR 1 // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2 // At least this minor version

namespace iree {

// Defined in iree_run.cpp.
void runModule(void *content, size_t size);

Compiler::Compiler() = default;

bool Compiler::init() {
  ireeCompilerGlobalInitialize();

  uint32_t apiVersion = (uint32_t)ireeCompilerGetAPIVersion();
  uint16_t apiVersionMajor = (uint16_t)((apiVersion >> 16) & 0xFFFFUL);
  uint16_t apiVersionMinor = (uint16_t)(apiVersion & 0xFFFFUL);
  if (apiVersionMajor > IREE_COMPILER_EXPECTED_API_MAJOR ||
      apiVersionMinor < IREE_COMPILER_EXPECTED_API_MINOR) {
    fmt::println(stderr,
                 "Error: incompatible API version; built for version {}.{} "
                 "but loaded version {}.{}",
                 IREE_COMPILER_EXPECTED_API_MAJOR,
                 IREE_COMPILER_EXPECTED_API_MINOR, apiVersionMajor,
                 apiVersionMinor);
    return false;
  }
  return true;
}

Compiler::~Compiler() { ireeCompilerGlobalShutdown(); }

std::unique_ptr<Compiler> Compiler::create() {
  auto compiler = std::unique_ptr<Compiler>(new Compiler());
  if (!compiler->init()) {
    return nullptr;
  }
  return compiler;
}

Error::Error() = default;
Error::Error(iree_compiler_error_t *error) : error(error) {}
Error::~Error() {
  if (error) {
    ireeCompilerErrorDestroy(error);
  }
}

Error::operator bool() const { return error != nullptr; }
const char *Error::getMessage() const {
  return ireeCompilerErrorGetMessage(error);
}

SourceWrapBuffer::~SourceWrapBuffer() {
  if (compilerSource) {
    ireeCompilerSourceDestroy(compilerSource);
  }
}

std::unique_ptr<Error> SourceWrapBuffer::init(iree_compiler_session_t *session,
                                              const std::string &source,
                                              const std::string &name) {
  auto error = ireeCompilerSourceWrapBuffer(
      session, name.c_str(), source.c_str(), source.length() + 1,
      /*isNullTerminated=*/true, &compilerSource);
  if (error) {
    return std::make_unique<Error>(error);
  }
  return nullptr;
}

CompilerInvocation::CompilerInvocation(iree_compiler_session_t *session) {
  invocation = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationEnableConsoleDiagnostics(invocation);
}

CompilerInvocation::~CompilerInvocation() {
  ireeCompilerInvocationDestroy(invocation);
}

bool CompilerInvocation::parseSource(SourceWrapBuffer *source) {
  return ireeCompilerInvocationParseSource(invocation, source->compilerSource);
}

bool CompilerInvocation::compile() {
  return ireeCompilerInvocationPipeline(invocation, IREE_COMPILER_PIPELINE_STD);
}

void CompilerInvocation::outputIR() {
  iree_compiler_output_t *output;
  ireeCompilerOutputOpenFD(fileno(stdout), &output);
  ireeCompilerInvocationOutputIR(invocation, output);
  ireeCompilerOutputDestroy(output);
}

void CompilerInvocation::run() {
  iree_compiler_output_t *output;
  ireeCompilerOutputOpenMembuffer(&output);
  ireeCompilerInvocationOutputVMBytecode(invocation, output);

  void *content;
  size_t size;
  ireeCompilerOutputMapMemory(output, &content, &size);

  runModule(content, size);

  ireeCompilerOutputDestroy(output);
}

CompilerSession::CompilerSession() {
  session = ireeCompilerSessionCreate();

  std::vector<const char *> flags = {
      "--iree-hal-target-backends=llvm-cpu",
  };

  ireeCompilerSetupGlobalCL(flags.size(), flags.data(), "iree_compile", false);

  auto error = ireeCompilerSessionSetFlags(session, flags.size(), flags.data());
  if (error) {
    fmt::println(stderr, "Error setting flags: {}",
                 ireeCompilerErrorGetMessage(error));
  }
}

CompilerSession::~CompilerSession() { ireeCompilerSessionDestroy(session); }

std::unique_ptr<SourceWrapBuffer>
CompilerSession::createSourceWrapBuffer(const std::string &source,
                                        const std::string &name) {
  auto sourceWrapper = std::make_unique<SourceWrapBuffer>();
  auto error = sourceWrapper->init(session, source, name);
  if (error) {
    fmt::println(stderr, "Error wrapping source buffer: {}",
                 error->getMessage());
    return nullptr;
  }
  return sourceWrapper;
}

std::unique_ptr<CompilerInvocation>
CompilerSession::createCompilerInvocation() {
  return std::unique_ptr<CompilerInvocation>(new CompilerInvocation(session));
}
} // namespace iree
