cmake_minimum_required(VERSION 3.16)
project(mmfreecpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# POSIX only (src/io/weights.cpp mmap()s the blob); on Windows, build under WSL.
if(WIN32)
  message(WARNING "mmfreecpu has not been ported to Win32 (src/io/weights.cpp uses mmap). "
                  "Build under WSL, or port load_blob() to MapViewOfFile.")
endif()

# Every flag below is GNU-style; other compilers (MSVC) only get a warning level.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  set(MMFREE_GNU_FLAGS ON)
else()
  set(MMFREE_GNU_FLAGS OFF)
endif()

# NO -ffast-math, here or in MMFREE_OPT_FLAGS: reordering fp reductions changes RMSNorm,
# hence quant_q510's int16 codes and the int32 accumulators, breaking the reference match
# and the CPU==FPGA gate.
# A cache variable so flags can be A/B'd without editing this file:
#   cmake -DMMFREE_OPT_FLAGS="-O3 -mcpu=cortex-a53" ...
# -O3 rather than -O2: GCC's -O2 vectorizer cost model skips runtime-trip-count loops.
# -mcpu/-mtune stay target-specific: the FPGA runner's Makefile or MMFREE_TUNE_NATIVE.
set(MMFREE_OPT_FLAGS "-O3" CACHE STRING "optimization flags for Release builds (no -ffast-math)")
if(MMFREE_GNU_FLAGS)
  set(CMAKE_CXX_FLAGS_RELEASE "${MMFREE_OPT_FLAGS}")
  add_compile_options(-Wall -Wextra)
else()
  add_compile_options(/W3)
endif()

# Host scheduling only (-mtune/-mcpu, not -march): no new instructions, numerics unchanged.
# OFF by default to keep builds reproducible across machines.
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

# SIMD backend, passed down as a compile define (see include/mmfree/simd.hpp):
#   AUTO (default) -> from CMAKE_SYSTEM_PROCESSOR; on x86, X86_512 if this host runs it
#   X86 -> AVX2 + FMA;  X86_512 -> AVX-512F/BW/DQ/VL + FMA (tq1.hpp's 512-bit paths);
#   ARM -> NEON;  SCALAR -> portable fallback (no define)
# AUTO is host-specific (even X86 needs AVX2); pass SCALAR or X86 for a portable binary.
set(MMFREE_SIMD "AUTO" CACHE STRING "SIMD backend: AUTO|X86|X86_512|ARM|SCALAR")
set_property(CACHE MMFREE_SIMD PROPERTY STRINGS AUTO X86 X86_512 ARM SCALAR)

if(MMFREE_SIMD STREQUAL "AUTO")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
    set(MMFREE_SIMD "X86")
    # Compile AND run the probe: compiling only proves the toolchain knows the intrinsics.
    # Skipped when cross-compiling; pass X86_512 explicitly there.
    if(NOT CMAKE_CROSSCOMPILING AND MMFREE_GNU_FLAGS)
      include(CheckCXXSourceRuns)
      set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma")
      # F + BW + VL: exactly what tq1.hpp gates MMFREE_AVX512 on.
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
  # PUBLIC: tq1.hpp/simd.hpp are header-only and gate on these flags, so consumers must
  # compile for the same ISA as the library.
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

# Threading: OpenMP, optional (without it the pragmas are ignored: correct but serial).
# macOS: AppleClang has no OpenMP runtime, Homebrew's libomp is keg-only (off the default
# search paths) and needs `-Xpreprocessor -fopenmp`, so seed FindOpenMP's hint variables
# BEFORE find_package (afterwards they are cached as not-found).
if(APPLE)
  # `brew --prefix libomp` first; the fixed prefixes cover a brew that is not on PATH.
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
  # Loud, since the only other symptom is single-core throughput.
  message(WARNING "OpenMP not found — building SINGLE-THREADED (correct, but much slower).\n"
                  "  macOS:  brew install libomp   (then re-run cmake; this file locates a\n"
                  "          keg-only libomp automatically. If it still misses, point at it:\n"
                  "          -DOpenMP_ROOT=$(brew --prefix libomp))\n"
                  "  Debian: sudo apt install libomp-dev  (clang)  /  g++ has it built in")
endif()

add_executable(mmfree-cli app/main.cpp)
target_link_libraries(mmfree-cli PRIVATE mmfreecpu)
add_executable(mmfree-chat app/chat.cpp)
target_link_libraries(mmfree-chat PRIVATE mmfreecpu)

include(GNUInstallDirs)
install(TARGETS mmfree-cli mmfree-chat mmfreecpu
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/mmfree DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

enable_testing()

# Packed-weight kernel self-test (no model blob needed). Links mmfreecpu to inherit its
# PUBLIC arch flags, so it tests the SIMD path the library was built for.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tq1.cpp)
  add_executable(test_tq1 tests/test_tq1.cpp)
  target_link_libraries(test_tq1 PRIVATE mmfreecpu)
  add_test(NAME tq1 COMMAND test_tq1)
endif()

# Golden-oracle tests from the `validation` branch; built only when their sources exist.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.cpp)
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE mmfreecpu)
  target_include_directories(test_kernels PRIVATE tests)
  # argv[2] = fixed-point golden dir
  add_test(NAME kernels
           COMMAND test_kernels ${CMAKE_CURRENT_SOURCE_DIR}/golden
                                ${CMAKE_CURRENT_SOURCE_DIR}/golden_fixedp)

  add_executable(test_e2e tests/test_e2e.cpp)
  target_link_libraries(test_e2e PRIVATE mmfreecpu)
  target_include_directories(test_e2e PRIVATE tests)
  add_test(NAME e2e
           COMMAND test_e2e ${CMAKE_CURRENT_SOURCE_DIR}/golden
                            ${CMAKE_CURRENT_SOURCE_DIR}/../blobs/370M/model.mmfree float)
  add_test(NAME e2e_fixed
           COMMAND test_e2e ${CMAKE_CURRENT_SOURCE_DIR}/golden_fixedp
                            ${CMAKE_CURRENT_SOURCE_DIR}/../blobs/370M/model.mmfree fixed)
endif()
