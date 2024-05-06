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

#define IREE_COMPILER_EXPECTED_API_MAJOR 1 // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2 // At least this minor version

namespace iree {

class Compiler {
public:
  ~Compiler();
  static std::unique_ptr<Compiler> create();

private:
  Compiler();
  bool init();
};

class Error {
public:
  Error();
  Error(iree_compiler_error_t *error);
  ~Error();
  explicit operator bool() const;
  const char *getMessage() const;

  Error(const Error &) = delete;
  Error &operator=(const Error &) = delete;
  Error(Error &&) = default;
  Error &operator=(Error &&) = default;

private:
  iree_compiler_error_t *error = nullptr;
};

class SourceWrapBuffer {
public:
  ~SourceWrapBuffer();
  friend class CompilerSession;
  friend class CompilerInvocation;

private:
  iree_compiler_source_t *compilerSource = nullptr;
  std::unique_ptr<Error> init(iree_compiler_session_t *session,
                              const std::string &source,
                              const std::string &name);
};

class CompilerInvocation {
public:
  ~CompilerInvocation();
  [[nodiscard]] bool parseSource(SourceWrapBuffer *source);
  [[nodiscard]] bool compile();
  void outputIR();
  void run();

private:
  iree_compiler_invocation_t *invocation;
  CompilerInvocation(iree_compiler_session_t *session);
  friend class CompilerSession;
};

class CompilerSession {
public:
  CompilerSession();
  ~CompilerSession();
  std::unique_ptr<SourceWrapBuffer>
  createSourceWrapBuffer(const std::string &source, const std::string &name);
  std::unique_ptr<CompilerInvocation> createCompilerInvocation();

private:
  iree_compiler_session_t *session;
};

} // namespace iree
