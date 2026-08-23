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
    src/nn/rmsnorm.cpp
    src/nn/embedding.cpp
    src/nn/RoPE.cpp
    src/nn/attention.cpp
    src/nn/mlp.cpp
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

add_executable(test_embedding tests/test_embedding.cpp)
target_include_directories(test_embedding PRIVATE tests)
target_link_libraries(test_embedding PRIVATE s1infer_core)

add_executable(test_rope tests/test_rope.cpp)
target_include_directories(test_rope PRIVATE tests)
target_link_libraries(test_rope PRIVATE s1infer_core)

add_executable(test_attention tests/test_attention.cpp)
target_include_directories(test_attention PRIVATE tests)
target_link_libraries(test_attention PRIVATE s1infer_core)

add_executable(test_mlp tests/test_mlp.cpp)
target_include_directories(test_mlp PRIVATE tests)
target_link_libraries(test_mlp PRIVATE s1infer_core)

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE s1infer_core)
target_compile_definitions(demo PRIVATE WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")
