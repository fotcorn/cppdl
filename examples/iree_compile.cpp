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

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

#include <fmt/core.h>

#define IREE_COMPILER_EXPECTED_API_MAJOR 1 // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2 // At least this minor version

typedef struct compiler_state_t {
  iree_compiler_session_t *session;
  iree_compiler_source_t *source;
  iree_compiler_output_t *output;
  iree_compiler_invocation_t *inv;
} compiler_state_t;

void handle_compiler_error(iree_compiler_error_t *error) {
  const char *msg = ireeCompilerErrorGetMessage(error);
  fprintf(stderr, "Error from compiler API:\n%s\n", msg);
  ireeCompilerErrorDestroy(error);
}

void cleanup_compiler_state(compiler_state_t s) {
  if (s.inv)
    ireeCompilerInvocationDestroy(s.inv);
  if (s.output)
    ireeCompilerOutputDestroy(s.output);
  if (s.source)
    ireeCompilerSourceDestroy(s.source);
  if (s.session)
    ireeCompilerSessionDestroy(s.session);
}

namespace iree {

class Compiler {
  Compiler() = default;

  bool init() {
    bool result = ireeCompilerLoadLibrary(CPPDL_IREE_COMPILER_LIB);
    if (!result) {
      fmt::println(stderr, "Failed to initialize IREE Compiler");
      return false;
    }
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

class CompilerSession {
  iree_compiler_session_t *session;

public:
  CompilerSession() { session = ireeCompilerSessionCreate(); }
  ~CompilerSession() { ireeCompilerSessionDestroy(session); }

  /*std::unique_ptr<SourceWrapBuffer>
  createSourceWrapBuffer(const std::string &source, const std::string &name,
                         Error &error) {
    return std::unique_ptr<SourceWrapBuffer>(
        new SourceWrapBuffer(session, source, name, error));
  }*/
};

// class SourceWrapBuffer {
//   iree_compiler_source_t *compilerSource;
//
//   SourceWrapBuffer(iree_compiler_session_t *session, const std::string
//   &source,
//                    const std::string &name, Error &error) {
//
//     auto compileError = ireeCompilerSourceWrapBuffer(
//         session, "simple_mul", source.c_str(), source.length() + 1,
//         /*isNullTerminated=*/true, &compilerSource);
//     // TODO Fail?
//   }
//
//   friend class CompilerSession;
// };

} // namespace iree

int main() {
  auto compiler = iree::Compiler::create();
  if (!compiler) {
    return 1;
  }

  iree::CompilerSession session;

  const char *simple_mul_mlir = " \
func.func @simple_mul(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {\n\
  %result = arith.mulf %lhs, %rhs : tensor<4xf32>\n \
  return %result : tensor<4xf32>\n \
}";

  /*
    iree::Error error;
    auto sourceBuffer = session.createSourceWrapBuffer(source, error);
    if (error) {
      fmt::println(stderr, "Error wrapping source buffer: {}",
                   error.getMessage());
      return 1;
    }*/
  iree_compiler_error_t *error;

  compiler_state_t s;

  error = ireeCompilerSourceWrapBuffer(s.session, "simple_mul", simple_mul_mlir,
                                       strlen(simple_mul_mlir) + 1,
                                       /*isNullTerminated=*/true, &s.source);
  if (error) {
    fprintf(stderr, "Error wrapping source buffer\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }
  fprintf(stdout, "Wrapped simple_mul buffer as compiler source\n");

  // ------------------------------------------------------------------------ //
  // Inputs and outputs are prepared, ready to run an invocation pipeline.    //
  // ------------------------------------------------------------------------ //

  // Use an invocation to compile from the input source to the output stream.
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(s.session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  if (!ireeCompilerInvocationParseSource(inv, s.source)) {
    fprintf(stderr, "Error parsing input source into invocation\n");
    cleanup_compiler_state(s);
    return 1;
  }

  // Compile up to the 'flow' dialect phase.
  // Typically a compiler tool would compile to 'end' and either output to a
  // .vmfb file for later usage in a deployed application or output to memory
  // for immediate usage in a JIT scenario.
  ireeCompilerInvocationSetCompileToPhase(inv, "flow");

  // Run the compiler invocation pipeline.
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    fprintf(stderr, "Error running compiler invocation\n");
    cleanup_compiler_state(s);
    return 1;
  }
  fprintf(stdout, "Compilation successful, output:\n\n");

  // Create a compiler 'output' piped to the 'stdout' file descriptor.
  // A file or memory buffer could be opened instead using
  // |ireeCompilerOutputOpenFile| or |ireeCompilerOutputOpenMembuffer|.
  fflush(stdout);
  error = ireeCompilerOutputOpenFD(fileno(stdout), &s.output);
  if (error) {
    fprintf(stderr, "Error opening output file descriptor\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }

  // Print IR to the output stream.
  // When compiling to the 'end' phase, a compiler tool would typically use
  // either |ireeCompilerInvocationOutputVMBytecode| or
  // |ireeCompilerInvocationOutputVMCSource|.
  error = ireeCompilerInvocationOutputIR(inv, s.output);
  if (error) {
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }

  cleanup_compiler_state(s);
  return 0;
}
