cmake_minimum_required(VERSION 3.16)
project(mmfreecpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Portability note: the engine targets POSIX (Linux, macOS, *BSD) on any CPU --
# src/io/weights.cpp mmap()s the weight blob. On Windows, build under WSL.
if(WIN32)
  message(WARNING "mmfreecpu has not been ported to Win32 (src/io/weights.cpp uses mmap). "
                  "Build under WSL, or port load_blob() to MapViewOfFile.")
endif()

# GCC/Clang vs MSVC: every flag below is a GNU-style one, so gate them on the compiler
# rather than assuming. MSVC needs a different spelling for all of them (and would fail
# the mmap check above first), so it only gets the warning level.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  set(MMFREE_GNU_FLAGS ON)
else()
  set(MMFREE_GNU_FLAGS OFF)
endif()

# NOTE: no -ffast-math -- it reorders fp reductions and breaks the "matches triton" check.
# (It would also change quant_q510's int16 output and therefore the engine's int32
# accumulators, breaking the CPU==FPGA gate. Keep it out of MMFREE_OPT_FLAGS too.)
#
# Optimization flags live in a CACHE variable so they can be A/B'd from the command line
# without editing this file — the kernels here are ~50% of FPGA-offload wall time, so the
# flags matter more than they look:
#   cmake -DMMFREE_OPT_FLAGS="-O3 -mcpu=cortex-a53" ...
# DEFAULT RAISED -O2 -> -O3 on 2026-07-30, measured not assumed. On the KRIA at 2.7B packed
# b=4, -O3 together with -mcpu=cortex-a53 cut host CPU time 12.8% and lifted decode 4.6%
# (29.56 -> 30.91 tok/s), with the CPU==FPGA gate PASSING on both arms — these flags keep IEEE
# fp semantics, so quant_q510's int16s and hence the engine's int32 accumulators are identical.
# Why -O2 was leaving throughput behind: it overrode CMake's own Release default of -O3, and at
# -O2 GCC 13 uses -fvect-cost-model=very-cheap, which declines loops whose trip count is only
# known at runtime -- e.g. the RMSNorm scale loop in bitlinear.cpp. -O3 switches to `dynamic`.
# -mcpu/-mtune are deliberately NOT set here: they are target-specific (the A53 is in-order
# dual-issue, so pipeline-tuned scheduling matters), so the FPGA runner's Makefile adds
# -mcpu=cortex-a53 on aarch64, MMFREE_TUNE_NATIVE below covers "build for this machine", and
# this file stays portable.
set(MMFREE_OPT_FLAGS "-O3" CACHE STRING "optimization flags for Release builds (no -ffast-math)")
if(MMFREE_GNU_FLAGS)
  set(CMAKE_CXX_FLAGS_RELEASE "${MMFREE_OPT_FLAGS}")
  add_compile_options(-Wall -Wextra)
else()
  add_compile_options(/W3)
endif()

# Schedule for the host CPU (-mtune/-mcpu, NOT -march): pure scheduling, it emits no new
# instructions, so the binary stays runnable on other CPUs of the same arch and the numerics
# are untouched. OFF by default to keep builds reproducible across machines.
option(MMFREE_TUNE_NATIVE "tune scheduling for the build host (-mtune=native / -mcpu=native)" OFF)

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
target_include_directories(mmfreecpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# SIMD backend: architecture-specific intrinsics selected at configure time and passed
# down as a compile define (see include/mmfree/simd.hpp). Override with -DMMFREE_SIMD=...
#   AUTO (default) -> detect from CMAKE_SYSTEM_PROCESSOR, and on x86 probe the host for
#                     AVX-512 (see below) so a machine that has it gets it by default
#   X86            -> AVX2 + FMA      (MMFREE_ARCH_X86)
#   X86_512        -> AVX-512F/BW/DQ/VL + FMA (MMFREE_ARCH_X86; tq1.hpp then compiles its
#                     512-bit ternary-dot paths, see MMFREE_AVX512 there). Host-specific:
#                     only select it on a machine that has AVX-512.
#   ARM            -> NEON            (MMFREE_ARCH_ARM)
#   SCALAR         -> portable fallback (no define)
#
# AUTO is deliberately HOST-SPECIFIC, not lowest-common-denominator: even plain X86 emits
# AVX2, which a pre-2013 CPU cannot run. For a binary that must run on an unknown machine,
# pass -DMMFREE_SIMD=SCALAR (or X86 for the AVX2 baseline) explicitly.
set(MMFREE_SIMD "AUTO" CACHE STRING "SIMD backend: AUTO|X86|X86_512|ARM|SCALAR")
set_property(CACHE MMFREE_SIMD PROPERTY STRINGS AUTO X86 X86_512 ARM SCALAR)

if(MMFREE_SIMD STREQUAL "AUTO")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
    set(MMFREE_SIMD "X86")
    # Prefer the 512-bit ternary-dot paths when the host can actually run them. This used
    # to need a manual -DMMFREE_SIMD=X86_512 and was worth ~29% on a Xeon, so "forgot the
    # flag" was a silent 1/3 of throughput. Compile AND run the probe: compiling only
    # proves the toolchain knows the intrinsics, not that this CPU executes them.
    # Skipped when cross-compiling (can't run host code for the target) -- pass X86_512
    # explicitly for a cross build.
    if(NOT CMAKE_CROSSCOMPILING AND MMFREE_GNU_FLAGS)
      include(CheckCXXSourceRuns)
      set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma")
      # The probe exercises F + BW + VL because that is exactly the trio tq1.hpp gates
      # MMFREE_AVX512 on; a CPU with F but not BW must fall back to AVX2.
      check_cxx_source_runs("
        #include <immintrin.h>
        int main() {
          __m512i a = _mm512_set1_epi16(3), b = _mm512_set1_epi16(5);
          __m512i c = _mm512_maddubs_epi16(a, b);                      // AVX512BW
          __m256i d = _mm256_maskz_add_epi32(0x0f, _mm512_castsi512_si256(c),
                                             _mm256_set1_epi32(1));    // AVX512VL
          return _mm512_reduce_add_epi32(_mm512_and_si512(               // AVX512F
                     _mm512_castsi256_si512(d), _mm512_set1_epi32(0)));
        }" MMFREE_HAVE_AVX512)
      unset(CMAKE_REQUIRED_FLAGS)
      if(MMFREE_HAVE_AVX512)
        set(MMFREE_SIMD "X86_512")
      endif()
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|armv8|armv7)")
    set(MMFREE_SIMD "ARM")
  else()
    set(MMFREE_SIMD "SCALAR")
  endif()
  message(STATUS "mmfree SIMD backend: ${MMFREE_SIMD} (auto-detected; override with -DMMFREE_SIMD=)")
else()
  message(STATUS "mmfree SIMD backend: ${MMFREE_SIMD} (requested)")
endif()

if(MMFREE_SIMD STREQUAL "X86")
  target_compile_definitions(mmfreecpu PUBLIC MMFREE_ARCH_X86)
  target_compile_options(mmfreecpu PUBLIC -mavx2 -mfma)
elseif(MMFREE_SIMD STREQUAL "X86_512")
  target_compile_definitions(mmfreecpu PUBLIC MMFREE_ARCH_X86)
  # PUBLIC, not PRIVATE: tq1.hpp is header-only and gates its 512-bit paths on __AVX512F__ /
  # __AVX512BW__, so a consumer compiled without these flags would silently get the AVX2 path
  # while this library got the 512-bit one -- the same incoherent-mix footgun the ARCH/SIMD
  # split above already warns about. (simd.hpp is header-only too, hence PUBLIC on X86.)
  target_compile_options(mmfreecpu PUBLIC -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
elseif(MMFREE_SIMD STREQUAL "ARM")
  target_compile_definitions(mmfreecpu PUBLIC MMFREE_ARCH_ARM)
  # AArch64 has NEON in the baseline ISA (no flag needed). 32-bit ARM needs -mfpu=neon.
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64)")
    target_compile_options(mmfreecpu PUBLIC -mfpu=neon)
  endif()
endif()

if(MMFREE_TUNE_NATIVE AND MMFREE_GNU_FLAGS)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|armv8|armv7)")
    set(MMFREE_TUNE_FLAG -mcpu=native)
  else()
    set(MMFREE_TUNE_FLAG -mtune=native)
  endif()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(${MMFREE_TUNE_FLAG} MMFREE_HAVE_TUNE_NATIVE)
  if(MMFREE_HAVE_TUNE_NATIVE)
    target_compile_options(mmfreecpu PRIVATE ${MMFREE_TUNE_FLAG})
    message(STATUS "mmfree host tuning: ${MMFREE_TUNE_FLAG}")
  else()
    message(WARNING "MMFREE_TUNE_NATIVE=ON but the compiler rejects ${MMFREE_TUNE_FLAG} — ignoring.")
  endif()
endif()

# Threading: parallelize the ternary matmul over output rows (see bitlinear.cpp).
# Optional -- without OpenMP the pragmas are ignored and the build stays single-threaded
# (and still correct, just slower; see include/mmfree/parallel.hpp).
#
# macOS needs help before we can even ask. AppleClang ships no OpenMP runtime, and
# `brew install libomp` alone does NOT make find_package(OpenMP) succeed, for two
# independent reasons:
#   1. libomp is KEG-ONLY -- Homebrew deliberately does not symlink it into
#      {/opt/homebrew,/usr/local}/{include,lib}, so the headers and dylib are not on any
#      default search path.
#   2. AppleClang rejects a bare `-fopenmp`; the OpenMP pragmas are only enabled through
#      `-Xpreprocessor -fopenmp`, which is not the spelling FindOpenMP tries first.
# So fill in the three hint variables FindOpenMP consults, and only then call it. Doing this
# BEFORE find_package matters -- afterwards they are already cached as not-found.
if(APPLE)
  # `brew --prefix libomp` is authoritative (handles Apple-silicon vs Intel prefixes and a
  # non-standard HOMEBREW_PREFIX); the hardcoded pair is the fallback when brew is not on
  # PATH, e.g. a CI runner with a bare Homebrew install.
  find_program(MMFREE_BREW brew)
  if(MMFREE_BREW)
    execute_process(COMMAND ${MMFREE_BREW} --prefix libomp
                    OUTPUT_VARIABLE MMFREE_BREW_LIBOMP
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  foreach(cand "${MMFREE_BREW_LIBOMP}" /opt/homebrew/opt/libomp /usr/local/opt/libomp)
    if(NOT "${cand}" STREQUAL "" AND NOT MMFREE_LIBOMP AND EXISTS "${cand}/include/omp.h")
      set(MMFREE_LIBOMP "${cand}")
    endif()
  endforeach()
  if(MMFREE_LIBOMP AND NOT DEFINED OpenMP_CXX_FLAGS)
    message(STATUS "mmfree: using Homebrew libomp at ${MMFREE_LIBOMP}")
    set(OpenMP_ROOT "${MMFREE_LIBOMP}")  # honoured by CMake >= 3.20
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${MMFREE_LIBOMP}/include")
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY "${MMFREE_LIBOMP}/lib/libomp.dylib")
  endif()
endif()
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mmfreecpu PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "mmfree threading: OpenMP ${OpenMP_CXX_VERSION}")
else()
  # Worth a loud line: the usual cause is AppleClang, which ships no OpenMP runtime, and
  # the only symptom otherwise is single-core throughput.
  message(WARNING "OpenMP not found — building SINGLE-THREADED (correct, but much slower).\n"
                  "  macOS:  brew install libomp   (then re-run cmake; this file locates a\n"
                  "          keg-only libomp automatically. If it still misses, point at it:\n"
                  "          -DOpenMP_ROOT=$(brew --prefix libomp))\n"
                  "  Debian: sudo apt install libomp-dev  (clang)  /  g++ has it built in")
endif()

add_executable(mmfree-cli app/main.cpp)
target_link_libraries(mmfree-cli PRIVATE mmfreecpu)

include(GNUInstallDirs)
install(TARGETS mmfree-cli mmfreecpu
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/mmfree DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

enable_testing()

# Kernel self-tests that need no model blob and no oracles — always available.
# test_tq1 links mmfreecpu purely to inherit its PUBLIC arch flags, so it exercises the
# same extraction path (AVX-512 / AVX2 / NEON / scalar) the library was built for.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tq1.cpp)
  add_executable(test_tq1 tests/test_tq1.cpp)
  target_link_libraries(test_tq1 PRIVATE mmfreecpu)
  add_test(NAME tq1 COMMAND test_tq1)
endif()

# The golden-oracle tests live on the `validation` branch (with the tooling in tools/ that
# generates them). They build only when their sources are present, so this same CMakeLists
# works on the lean `main` branch and on `validation`.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.cpp)
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
