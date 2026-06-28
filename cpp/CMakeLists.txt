cmake_minimum_required(VERSION 3.16)
project(mmfreecpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# NOTE: no -ffast-math -- it reorders fp reductions and breaks the "matches triton" check.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra)

add_library(mmfreecpu STATIC
  src/kernels/rmsnorm.cpp
  src/kernels/bitlinear.cpp
  src/kernels/activations.cpp
  src/kernels/hgrn_scan.cpp
  src/kernels/rmsnorm_swishgate.cpp
  src/io/weights.cpp
  src/io/tokenizer.cpp
  src/model/hgrn_model.cpp
  src/util/profile.cpp
  src/util/bench.cpp
)
target_include_directories(mmfreecpu PUBLIC include)

# SIMD backend: architecture-specific intrinsics selected at configure time and passed
# down as a compile define (see include/mmfree/simd.hpp). Override with -DMMFREE_SIMD=...
#   AUTO (default) -> detect from CMAKE_SYSTEM_PROCESSOR
#   X86            -> AVX2 + FMA      (MMFREE_ARCH_X86)
#   ARM            -> NEON            (MMFREE_ARCH_ARM)
#   SCALAR         -> portable fallback (no define)
set(MMFREE_SIMD "AUTO" CACHE STRING "SIMD backend: AUTO|X86|ARM|SCALAR")
if(MMFREE_SIMD STREQUAL "AUTO")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
    set(MMFREE_SIMD "X86")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|armv8)")
    set(MMFREE_SIMD "ARM")
  else()
    set(MMFREE_SIMD "SCALAR")
  endif()
endif()
message(STATUS "mmfree SIMD backend: ${MMFREE_SIMD} (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")

if(MMFREE_SIMD STREQUAL "X86")
  target_compile_definitions(mmfreecpu PUBLIC MMFREE_ARCH_X86)
  target_compile_options(mmfreecpu PRIVATE -mavx2 -mfma)
elseif(MMFREE_SIMD STREQUAL "ARM")
  target_compile_definitions(mmfreecpu PUBLIC MMFREE_ARCH_ARM)
  # AArch64 has NEON in the baseline ISA (no flag needed). 32-bit ARM needs -mfpu=neon.
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64)")
    target_compile_options(mmfreecpu PRIVATE -mfpu=neon)
  endif()
endif()

# Threading: parallelize the ternary matmul over output rows (see bitlinear.cpp).
# Optional -- without OpenMP the pragmas are ignored and the build stays single-threaded.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mmfreecpu PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(mmfree-cli app/main.cpp)
target_link_libraries(mmfree-cli PRIVATE mmfreecpu)

# Validation tests live on the `validation` branch (with the golden-oracle tooling
# in tools/). They build only when tests/ is present, so this same CMakeLists works
# on the lean `main` branch (engine only) and on `validation` (engine + tests).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.cpp)
  enable_testing()
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE mmfreecpu)
  target_include_directories(test_kernels PRIVATE tests)
  # run from the cpp/ source dir so the default golden/ path resolves; argv[2] = fixed-point golden
  add_test(NAME kernels
           COMMAND test_kernels ${CMAKE_CURRENT_SOURCE_DIR}/golden
                                ${CMAKE_CURRENT_SOURCE_DIR}/golden_fixedp)

  add_executable(test_e2e tests/test_e2e.cpp)
  target_link_libraries(test_e2e PRIVATE mmfreecpu)
  target_include_directories(test_e2e PRIVATE tests)
  add_test(NAME e2e
           COMMAND test_e2e ${CMAKE_CURRENT_SOURCE_DIR}/golden
                            ${CMAKE_CURRENT_SOURCE_DIR}/model.mmfree float)
  add_test(NAME e2e_fixed
           COMMAND test_e2e ${CMAKE_CURRENT_SOURCE_DIR}/golden_fixedp
                            ${CMAKE_CURRENT_SOURCE_DIR}/model.mmfree fixed)
endif()
