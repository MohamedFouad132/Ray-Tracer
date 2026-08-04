// cpu_render.cu
// This file contains the CPU implementation of the ray tracing rendering function.
// It performs the same rendering operations as the GPU version but does so sequentially on the CPU.


#include <random>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "intersect.cuh"
#include "render.cuh"

// CPU rendering function that performs the equivalent work sequentially on the CPU.
void cpu_render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length){

    // Iterate over every pixel sequentially, row by row
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            int pixel_index = y * width + x;

            // Use a standard C++ random number generator for anti-aliasing
            // Seed it with the pixel index to ensure different random values for each pixel
            std::mt19937 rng(pixel_index);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            vec3 pixel_color(0.0f, 0.0f, 0.0f);

            // For each sample, jitter the ray direction slightly to achieve anti-aliasing.
            for (int s = 0; s < SAMPLES_PER_PIXEL; s++){
                float jitter_x = dist(rng);
                float jitter_y = dist(rng);

                float u = (float(x) + jitter_x) / (width - 1);
                float v = (float(y) + jitter_y) / (height - 1);

                float viewport_x = (u-0.5f) * viewport_width;
                float viewport_z = (0.5f-v) * viewport_height + camera_origin.z; // Adjusted to move the viewport in front of the camera

                // Compute ray direction and create a ray from the camera origin through the computed viewport point
                vec3 viewport_point(viewport_x, camera_origin.y + focal_length, viewport_z);
                vec3 ray_direction = (viewport_point - camera_origin).normalize();
                ray r(camera_origin, ray_direction);

                // Trace the ray and accumulate the color contribution from this sample
                pixel_color = pixel_color + trace_ray(r, spheres, num_spheres, light_position);
            }
            // Average the accumulated color samples to get the final pixel color and store it in the framebuffer
            pixel_color = pixel_color * (1.0f / SAMPLES_PER_PIXEL);
            framebuffer[pixel_index] = pixel_color;
        }
    }
}