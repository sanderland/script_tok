# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
#cmake_policy(SET CMP0148 NEW)
project(my_fast_tokenizer LANGUAGES CXX)

# Find dependencies
include(FetchContent)

# Find pybind11 and Python
find_package(pybind11 REQUIRED)

# Find Abseil on the system
find_package(absl REQUIRED COMPONENTS hash raw_hash_set strings)

# Find Boost
find_package(Boost REQUIRED)

# Add GTL
FetchContent_Declare(
    gtl
    GIT_REPOSITORY https://github.com/greg7mdp/gtl.git
    GIT_TAG        v1.2.0
)

# Add robin_hood
FetchContent_Declare(
    robin_hood
    GIT_REPOSITORY https://github.com/martinus/robin-hood-hashing.git
    GIT_TAG 3.11.5
)

# Add ska::flat_hash_map
FetchContent_Declare(
    ska
    GIT_REPOSITORY https://github.com/skarupke/flat_hash_map.git
    GIT_TAG master  # ska has no tags, using master
)

# Make all dependencies available
FetchContent_MakeAvailable(gtl robin_hood ska)

# Note: It's better to use CMake's check_cxx_compiler_flag for portability
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_HAS_MARCH_NATIVE)

# Function to set common compile options
function(set_common_target_options target implementation queue_type)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE 
        -O3 
        -DNDEBUG
        -g
        -Wno-interference-size
    )
    if(COMPILER_HAS_MARCH_NATIVE)
        target_compile_options(${target} PRIVATE -march=native)
    endif()

    # Enable parallel LTO
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_target_properties(${target} PROPERTIES 
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
        target_compile_options(${target} PRIVATE -flto=auto)
    endif()

    # Add common includes
    target_include_directories(${target} PRIVATE 
        ${gtl_SOURCE_DIR}/include
        ${robin_hood_SOURCE_DIR}/src/include
        ${ska_SOURCE_DIR}
    )

    # Set implementation-specific options
    if(implementation STREQUAL "std")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_STD)
    elseif(implementation STREQUAL "absl")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_ABSL)
        target_link_libraries(${target} PRIVATE absl::flat_hash_map)
    elseif(implementation STREQUAL "gtl")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_GTL)
    elseif(implementation STREQUAL "robinhood")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_ROBINHOOD)
    elseif(implementation STREQUAL "ska")
        target_compile_definitions(${target} PRIVATE MERGE_MAP_SKA)
    endif()

    target_compile_definitions(${target} PRIVATE ${queue_type})
endfunction()

# Function to create a single target with specific implementation and queue type
function(create_specific_target target_base source_files implementation queue_type suffix)
    set(target_name "${target_base}${suffix}")
    if(target_base STREQUAL "fast_tokenizer_cpp")
        pybind11_add_module(${target_name} ${source_files})
        set_target_properties(${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/script_bpe/fast"
            OUTPUT_NAME "${target_name}"
        )
    else()
        add_executable(${target_name} ${source_files})
        target_compile_definitions(${target_name} PRIVATE NO_PYTHON_BINDINGS)
    endif()
    set_common_target_options(${target_name} ${implementation} ${queue_type})
endfunction()

# Create BPE Python module - only GTL + PQ_BOOST_4ARY variant
create_specific_target(
    fast_tokenizer_cpp
    "script_bpe/fast/python_bindings.cpp;script_bpe/fast/bpe_core.cpp"
    "gtl"
    "PQ_BOOST_4ARY"
    ""
)

# Create Unigram Python module
pybind11_add_module(fast_unigram_cpp 
    "script_bpe/fast/unigram_bindings.cpp"
    "script_bpe/fast/unigram_core.cpp"
)
target_compile_features(fast_unigram_cpp PRIVATE cxx_std_17)
target_compile_options(fast_unigram_cpp PRIVATE -O3 -DNDEBUG)
set_target_properties(fast_unigram_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/script_bpe/fast"
    OUTPUT_NAME "fast_unigram_cpp"
)

# Create benchmark executables with specific variants
create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "gtl" "PQ_STD_HEAP" "__gtl_stdheap")
create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "gtl" "PQ_BOOST_4ARY" "__gtl_boost4ary")

create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "absl" "PQ_STD_HEAP" "__absl_stdheap")
create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "absl" "PQ_BOOST_4ARY" "__absl_boost4ary")

create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "ska" "PQ_STD_HEAP" "__ska_stdheap")
create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "ska" "PQ_BOOST_4ARY" "__ska_boost4ary")

create_specific_target(benchmark_cpp_fast "script_bpe/fast/benchmark_cpp_fast.cpp;script_bpe/fast/bpe_core.cpp" "std" "PQ_STD_HEAP" "__std_stdheap")
