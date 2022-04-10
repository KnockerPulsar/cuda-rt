#include <iostream>

__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  std::cout << "Launching kernel\n";

  int blockSize = 256;
  //                 v----------v this part is for rounding up
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  cudaDeviceSynchronize();

  std::cout << "Kernel done\n";

  cudaFree(x);
  cudaFree(y);

  std::cout << "Memory freed\n";

  return 0;
}