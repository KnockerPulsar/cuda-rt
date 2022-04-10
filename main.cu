#include <iostream>

__global__ void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++)
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
  add<<<1,1>>>(N, x, y);

  cudaDeviceSynchronize();

  std::cout << "Kernel done\n";

  cudaFree(x);
  cudaFree(y);

  std::cout << "Memory freed\n";

  return 0;
}