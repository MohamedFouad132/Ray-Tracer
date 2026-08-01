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


int main(int argc, char** argv){

    // Parse command line arguments for image width, height and rendering mode
    int width = 1920;
    int height = 1080;
    std::string mode = "gpu"; // Default mode is GPU
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Handle --help or -h flag to display usage information
        if (arg == "--help" || arg == "-h") {
            printf("Usage: raytracer [options]\n");
            printf("  --width <n>     Image width in pixels (default: 1920)\n");
            printf("  --height <n>    Image height in pixels (default: 1080)\n");
            printf("  --mode <mode>   Render mode: cpu, gpu, gpu-optimized (default: gpu)\n");
            printf("  --help, -h      Show this message\n");
            return 0;
        
        // Handle --width flag to set image width
        } else if (arg == "--width"){
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --width requires a value\n");
                return 1;
            }

            width = std::stoi(argv[++i]);
            if (width <= 0) {
                fprintf(stderr, "Error: Width must be a positive integer\n");
                return 1;
            }

        // Handle --height flag to set image height
        } else if (arg == "--height") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --height requires a value\n");
                return 1;
            }

            height = std::stoi(argv[++i]);
            if (height <= 0) {
                fprintf(stderr, "Error: Height must be a positive integer\n");
                return 1;
            }

        // Handle --mode flag to set rendering mode (cpu, gpu, gpu-optimized)
        } else if (arg == "--mode") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --mode requires a value\n");
                return 1;
            }

            mode = argv[++i];
            if (mode != "cpu" && mode != "gpu" && mode != "gpu-optimized") {
                fprintf(stderr, "Error: Invalid mode. Use 'cpu', 'gpu', or 'gpu-optimized'\n");
                return 1;
            }
        }

        // Handle unknown flags
        else {
            fprintf(stderr, "Error: Unknown flag '%s'\n", arg.c_str());
            fprintf(stderr, "Use --help or -h for usage information.\n");
            return 1;
        }
    }


    // Define constants for the viewport dimensions and focal length based on the image aspect ratio
    const float VIEWPORT_HEIGHT = 2.0f;
    const float VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (float(width) / height);
    const float FOCAL_LENGTH = 1.5f;

    
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
    int num_pixels = width * height;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3* h_framebuffer = (vec3*)malloc(framebuffer_size);
    
    if (mode == "cpu") {
        // If the user specified the --cpu flag, use the CPU rendering function
        cpu_render(h_framebuffer, camera_origin, h_spheres, num_spheres, light_position, width, height, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);

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
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Launch the appropriate GPU rendering kernel based on the specified mode (gpu or gpu-optimized)
        if (mode == "gpu-optimized") {
            gpu_render_optimized<<<gridDim, blockDim>>>(d_framebuffer, camera_origin, d_spheres, num_spheres, light_position, width, height, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);
        } else {
            gpu_render<<<gridDim, blockDim>>>(d_framebuffer, camera_origin, d_spheres, num_spheres, light_position, width, height, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);
        }
        // Wait for the GPU to finish before copying the framebuffer back to the host
        cudaDeviceSynchronize();
        cudaMemcpy(h_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost);
        // Free the allocated GPU memory for the framebuffer and spheres
        cudaFree(d_framebuffer);
        cudaFree(d_spheres);
    }

    // Write the framebuffer to a PPM file (a text based image format)
    FILE* f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    
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