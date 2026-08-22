cmake_minimum_required(VERSION 3.24)
project(s1infer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_library(s1infer_core STATIC
    src/tensor.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/rmsnorm.cpp
)
target_include_directories(s1infer_core PUBLIC include)

add_executable(test_m0 tests/test_m0_tensor.cpp)
target_include_directories(test_m0 PRIVATE tests)
target_link_libraries(test_m0 PRIVATE s1infer_core)

add_executable(test_m1 tests/test_m1_safetensors.cpp)
target_include_directories(test_m1 PRIVATE tests)
target_link_libraries(test_m1 PRIVATE s1infer_core)
target_compile_definitions(test_m1 PRIVATE WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

add_executable(test_bpe tests/test_bpe_tokenizer.cpp)
target_include_directories(test_bpe PRIVATE tests)
target_link_libraries(test_bpe PRIVATE s1infer_core)
target_compile_definitions(test_bpe PRIVATE WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
target_include_directories(test_rmsnorm PRIVATE tests)
target_link_libraries(test_rmsnorm PRIVATE s1infer_core)
