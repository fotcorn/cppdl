#pragma once

#include <chrono>
#include <fmt/core.h>
#include <string>

class Timer {
public:
  Timer(const std::string &scope)
      : start(std::chrono::high_resolution_clock::now()), lastScope(scope) {}

  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    fmt::println("Scope \"{}\" took {} ms", lastScope, duration.count());
  }

  void time(const std::string &scope) {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = now - start;
    fmt::println("Scope \"{}\" took {} ms", lastScope, duration.count());
    lastScope = scope;
    start = now;
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::string lastScope;
};
