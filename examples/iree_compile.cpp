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

#include "iree_run.cpp"

#define IREE_COMPILER_EXPECTED_API_MAJOR 1 // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2 // At least this minor version

namespace iree {

class Compiler {
  Compiler() = default;

  bool init() {
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

public:
  ~Compiler() { ireeCompilerGlobalShutdown(); }

  static std::unique_ptr<Compiler> create() {
    auto compiler = std::unique_ptr<Compiler>(new Compiler());
    if (!compiler->init()) {
      return nullptr;
    }
    return compiler;
  }
};

class Error {
  iree_compiler_error_t *error = nullptr;

public:
  Error() = default;
  Error(iree_compiler_error_t *error) : error(error) {}
  Error(const Error &) = delete;
  Error &operator=(const Error &) = delete;
  Error(Error &&) = default;
  Error &operator=(Error &&) = default;

  explicit operator bool() const { return error != nullptr; }
  const char *getMessage() const { return ireeCompilerErrorGetMessage(error); }

  ~Error() {
    if (error) {
      ireeCompilerErrorDestroy(error);
    }
  }
};

class SourceWrapBuffer {
  iree_compiler_source_t *compilerSource = nullptr;
  std::unique_ptr<Error> init(iree_compiler_session_t *session,
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

public:
  ~SourceWrapBuffer() {
    if (compilerSource) {
      ireeCompilerSourceDestroy(compilerSource);
    }
  }
  friend class CompilerSession;
  friend class CompilerInvocation;
};

class CompilerInvocation {
  iree_compiler_invocation_t *invocation;

  CompilerInvocation(iree_compiler_session_t *session) {
    invocation = ireeCompilerInvocationCreate(session);
    ireeCompilerInvocationEnableConsoleDiagnostics(invocation);
  }

public:
  [[nodiscard]] bool parseSource(SourceWrapBuffer *source) {
    return ireeCompilerInvocationParseSource(invocation,
                                             source->compilerSource);
  }

  [[nodiscard]] bool compile() {
    return ireeCompilerInvocationPipeline(invocation,
                                          IREE_COMPILER_PIPELINE_STD);
  }

  void outputIR() {
    iree_compiler_output_t *output;
    ireeCompilerOutputOpenFD(fileno(stdout), &output);
    ireeCompilerInvocationOutputIR(invocation, output);
    ireeCompilerOutputDestroy(output);
  }

  void run() {
    iree_compiler_output_t *output;
    ireeCompilerOutputOpenMembuffer(&output);
    ireeCompilerInvocationOutputVMBytecode(invocation, output);

    void *content;
    size_t size;
    ireeCompilerOutputMapMemory(output, &content, &size);

    runModule(content, size);

    ireeCompilerOutputDestroy(output);
  }

  ~CompilerInvocation() { ireeCompilerInvocationDestroy(invocation); }

  friend class CompilerSession;
};

class CompilerSession {
  iree_compiler_session_t *session;

public:
  CompilerSession() {
    session = ireeCompilerSessionCreate();

    std::vector<const char *> flags = {
        //"--iree-hal-device-target=llvm-cpu",
        "--iree-hal-target-backends=llvm-cpu",
        //"--iree-hal-executable-target=embedded-elf-x86_64",
    };

    ireeCompilerSetupGlobalCL(flags.size(), flags.data(), "iree_compile",
                              false);

    auto error =
        ireeCompilerSessionSetFlags(session, flags.size(), flags.data());
    if (error) {
      fmt::println(stderr, "Error setting flags: {}",
                   ireeCompilerErrorGetMessage(error));
    }
  }
  ~CompilerSession() { ireeCompilerSessionDestroy(session); }

  std::unique_ptr<SourceWrapBuffer>
  createSourceWrapBuffer(const std::string &source, const std::string &name) {
    auto sourceWrapper = std::make_unique<SourceWrapBuffer>();
    auto error = sourceWrapper->init(session, source, name);
    if (error) {
      fmt::println(stderr, "Error wrapping source buffer: {}",
                   error->getMessage());
      return nullptr;
    }
    return sourceWrapper;
  }

  std::unique_ptr<CompilerInvocation> createCompilerInvocation() {
    return std::unique_ptr<CompilerInvocation>(new CompilerInvocation(session));
  }
};

} // namespace iree

int main() {
  auto compiler = iree::Compiler::create();
  if (!compiler) {
    return 1;
  }

  iree::CompilerSession session;

  std::string simple_mul_mlir = " \
func.func @simple_mul(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {\n\
  %result = arith.mulf %lhs, %rhs : tensor<4xf32>\n \
  return %result : tensor<4xf32>\n \
}";

  auto sourceWrapBuffer =
      session.createSourceWrapBuffer(simple_mul_mlir, "simple_mul");

  if (!sourceWrapBuffer) {
    return 1;
  }

  auto invocation = session.createCompilerInvocation();

  if (!invocation->parseSource(sourceWrapBuffer.get())) {
    fmt::println(stderr, "Error parsing source.");
    return 1;
  }

  if (!invocation->compile()) {
    fmt::println(stderr, "Error compiling source.");
    return 1;
  }

  invocation->run();

  //invocation->outputIR();

  return 0;
}
