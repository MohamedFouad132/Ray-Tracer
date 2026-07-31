// main.cu
// This is the main entry point for the ray tracing application.
// It sets up the scene, initializes the camera and light source, and renders the image using either the GPU (default) or the CPU (--cpu flag).
// Result is saved as a PPM file.


// Include defined headers
#include <cstdio>
#include <string>
#include "vec3.cuh"
#include "sphere.cuh"
#include "render.cuh"


const int IMAGE_WIDTH = 1920;
const int IMAGE_HEIGHT = 1080;
const float VIEWPORT_HEIGHT = 2.0f;
const float VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (float(IMAGE_WIDTH) / IMAGE_HEIGHT); // This is computed to preserve the aspect ratio of the image.
const float FOCAL_LENGTH = 1.5f;

int main(int argc, char** argv){
    bool use_cpu = (argc > 1 && std::string(argv[1]) == "--cpu");


    // Define the scene with spheres, light source and camera position
    int num_spheres = 4;
    sphere h_spheres[4] = {
        sphere(vec3(-4.5f, 5.0f, 0.7f), 0.6f, vec3(0.0f, 0.0f, 1.0f), 0.0f),
        sphere(vec3(-1.9f, 5.0f, 1.1f), 1.0f, vec3(1.0f, 0.0f, 0.0f), 0.0f),
        sphere(vec3( 1.1f, 5.0f, 1.1f), 1.0f, vec3(0.9f, 0.9f, 0.9f), 0.82f),
        sphere(vec3( 3.9f, 5.0f, 0.9f), 0.8f, vec3(1.0f, 1.0f, 0.0f), 0.0f)
    };

    vec3 light_position(-1.0f, 3.0f, 10.0f);
    vec3 camera_origin(0.0f, -0.5f, 2.0f);

    // Allocate memory on the host for the framebuffer
    int num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3* h_framebuffer = (vec3*)malloc(framebuffer_size);
    
    if (use_cpu) {
        // If the user specified the --cpu flag, use the CPU rendering function
        cpu_render(h_framebuffer, camera_origin, h_spheres, num_spheres, light_position, IMAGE_WIDTH, IMAGE_HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);

    } else {
        // Otherwise, use the GPU rendering function

        // Allocate memory on the GPU for the spheres and copy the sphere data from host to device
        sphere* d_spheres;
        cudaMalloc(&d_spheres, num_spheres * sizeof(sphere));
        cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(sphere), cudaMemcpyHostToDevice);
        // Allocate memory on the GPU for the framebuffer
        vec3 *d_framebuffer;
        cudaMalloc(&d_framebuffer, framebuffer_size);

        // Define the block and grid dimensions for launching the CUDA kernel. Each block contains 16x16 threads.
        // The formula for grid dimensions applies ceiling division to ensure that all pixels are covered even if the image dimensions are not multiples of the block size.
        dim3 blockDim(16, 16);
        dim3 gridDim((IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
        gpu_render<<<gridDim, blockDim>>>(d_framebuffer, camera_origin, d_spheres, num_spheres, light_position, IMAGE_WIDTH, IMAGE_HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);
        // Wait for the GPU to finish before copying the framebuffer back to the host
        cudaDeviceSynchronize();
        cudaMemcpy(h_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost);
        // Free the allocated GPU memory for the framebuffer and spheres
        cudaFree(d_framebuffer);
        cudaFree(d_spheres);
    }

    // Write the framebuffer to a PPM file (a text based image format)
    FILE* f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Define a lambda function to clamp color values between 0.0 and 1.0 before converting to 0-255 range.
    auto clamp = [](float v){ return fmin(1.0f, fmax(0.0f, v)); };
    
    for (int i = 0; i < num_pixels; i++) {
        vec3 color = h_framebuffer[i];
        int r = int(clamp(color.x) * 255);
        int g = int(clamp(color.y) * 255);
        int b = int(clamp(color.z) * 255);
        fprintf(f, "%d %d %d\n", r, g, b);
    }

    fclose(f);

    // Free allocated memory on host
    free(h_framebuffer);
    return 0;
}