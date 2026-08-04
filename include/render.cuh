// render.cuh
// Header file for render.cu declaring the GPU and CPU rendering functions.

#pragma once
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

inline const int SAMPLES_PER_PIXEL = 4;

__host__ __device__ vec3 trace_ray(ray r, sphere* spheres, int num_spheres, vec3 light_position);
__host__ __device__ vec3 trace_ray_optimized(ray r, sphere* spheres, int num_spheres, vec3 light_position);

__global__ void gpu_render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length);

__global__ void gpu_render_optimized(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length);

void cpu_render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length);