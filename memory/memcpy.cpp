#include <cstdint>
#include <iostream>

#include "common/timer.hpp"
#include <thrust/device_vector.h>

template <typename T>
__global__ void memcpy_kernel(const T *input, T *output, size_t size) {
  std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    output[tid] = input[tid];
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <num_elements> <num_experiments>"
              << std::endl;
    return 1;
  }

  size_t num_elements = std::atoll(argv[1]);
  size_t num_experiments = std::atoll(argv[2]);

  using T = uint32_t;
  thrust::device_vector<T> input(num_elements * 2, 1);

  gpu_timer timer;
  timer.start();
  for (size_t exp = 0; exp < num_experiments; ++exp) {
    std::uint32_t block_size = 256;
    std::uint32_t num_blocks = (num_elements + block_size - 1) / block_size;
    memcpy_kernel<<<num_blocks, block_size>>>(
        input.data().get(), input.data().get() + num_elements, num_elements);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
      return 1;
    }
  }
  timer.stop();

  double average_seconds = static_cast<double>(timer.get_elapsed_s()) /
                           static_cast<double>(num_experiments);

  double to_gibs = static_cast<double>(1ull << 30);
  double gigabytes =
      static_cast<double>(num_elements) * sizeof(T) * 2. / to_gibs;

  std::cout << "GiB/s " << gigabytes / average_seconds << std::endl;
  return 0;
}
