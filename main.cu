#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <curand_kernel.h>
#include <float.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ bool hit_sphere(const vec3 &center, float radius, const ray &r) {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0f * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4.0f * a * c;
  return (discriminant > 0.0f);
}

__device__ vec3 color(const ray &r, hitable **world) {
  hit_record rec;
  if ((*world)->hit(r, 0.0, FLT_MAX, rec))
    return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f,
                       rec.normal.z() + 1.0f);
  else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
  }
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns,
                       camera **cam, hitable **world, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= max_x || j >= max_y)
    return;

  int pixel_index = j * max_x + i;

  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);

  for (int s = 0; s < ns ; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v);
    col += color(r, world);
  }

  fb[pixel_index] = col / float(ns);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera** d_camera) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_list[0] = new sphere(vec3(0, 0, -1), 0.5);
    d_list[1] = new sphere(vec3(0, -100.5, -1), 100);
    *d_world = new hitable_list(d_list, 2);
    *d_camera = new camera();
  }
}

__global__ void free_world(hitable **d_list, hitable **d_world,
                           camera **d_camera) {
  delete *(d_list);
  delete *(d_list + 1);
  delete *d_world;
  delete *d_camera;
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= max_x || j >= max_y)
    return;

  int pixel_index = j * max_x + i;
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

int main(void) {

  int tx = 32, ty = 32;
  int nx = 1920, ny = 1080, ns = 100;

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  hitable **d_list; // d_* marks a GPU only variable
  checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hitable *)));

  hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

  curandState *d_rand_state;
  checkCudaErrors(
      cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

  create_world<<<1, 1>>>(d_list, d_world, d_camera);
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  free_world<<<1, 1>>>(d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_camera));

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      int ir = int(255.99 * fb[pixel_index].r());
      int ig = int(255.99 * fb[pixel_index].g());
      int ib = int(255.99 * fb[pixel_index].b());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  checkCudaErrors(cudaFree(fb));
}