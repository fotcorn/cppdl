# Compile
```
mkdir build
cd build
cmake -GNinja .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=clang-18 \
    -DCMAKE_CXX_COMPILER=clang++-18 \
    -DMLIR_DIR=<path to mlir install>/lib/cmake/mlir
```
