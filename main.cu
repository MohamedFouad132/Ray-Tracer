#include <cstdio>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "intersect.cuh"

const int IMAGE_WIDTH = 800;
const int IMAGE_HEIGHT = 600;

const float VIEWPORT_HEIGHT = 2.0f;

// preserving the aspect ratio of the image
const float VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (float(IMAGE_WIDTH) / IMAGE_HEIGHT);
const float FOCAL_LENGTH = 1.0f;


__global__ void render(vec3* framebuffer, vec3 camera_origin, sphere s, int width, int height, float viewport_width, float viewport_height, float focal_length) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = float(x) / (width - 1);
    float v = float(y) / (height - 1);

    float viewport_x = (u-0.5f) * viewport_width;
    float viewport_y = (0.5f-v) * viewport_height;

    vec3 viewport_point(viewport_x, viewport_y, -focal_length);
    vec3 ray_direction = (viewport_point - camera_origin).normalize();

    ray r(camera_origin, ray_direction);
    float t;
    int pixel_index = y * width + x;

    if (hit_sphere(r, s, t)){
        framebuffer[pixel_index] = s.color;
    } else {
        framebuffer[pixel_index] = vec3(0.0f, 0.0f, 0.0f);
    }


}


int main(){

    sphere s(vec3(0.0f, 0.0f, -2.0f), 0.5f, vec3(1.0f, 0.0f, 0.0f));
    int num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3 *d_framebuffer;
    cudaMalloc(&d_framebuffer, framebuffer_size);
    vec3 camera_origin(0.0f, 0.0f, 0.0f);

    dim3 blockDim(16, 16);
    dim3 gridDim((IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
    render<<<gridDim, blockDim>>>(d_framebuffer, camera_origin, s, IMAGE_WIDTH, IMAGE_HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);
    cudaDeviceSynchronize();
    
    vec3* h_framebuffer = (vec3*)malloc(framebuffer_size);
    cudaMemcpy(h_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost);
    FILE* f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);

    for (int i = 0; i < num_pixels; i++) {
        vec3 color = h_framebuffer[i];
        // ray directions range roughly -1 to 1, remap to 0-255 for a visible image
        int r = int(color.x * 255);
        int g = int(color.y * 255);
        int b = int(color.z * 255);
        fprintf(f, "%d %d %d\n", r, g, b);
    }

    fclose(f);


    free(h_framebuffer);
    cudaFree(d_framebuffer);
    return 0;
}

