cmake_minimum_required(VERSION 3.16)
project(BinaryFormer)

set(CMAKE_CXX_STANDARD 17)
enable_language(OBJCXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(BinaryFormer hammingAttention.mm main.cpp)
target_link_libraries(BinaryFormer "${TORCH_LIBRARIES}")

target_link_libraries(BinaryFormer "-framework Foundation")
target_link_libraries(BinaryFormer "-framework Metal")
target_link_libraries(BinaryFormer "-framework MetalPerformanceShaders")
target_link_libraries(BinaryFormer "-framework MetalPerformanceShadersGraph")

