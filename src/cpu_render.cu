// cpu_render.cu
// This file contains the multithreaded CPU implementation of the ray tracing rendering function.
// Threads fetch and render rows using an atomic counter until none remain .


#include <random>
#include <thread>
#include <vector>
#include <atomic>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "intersect.cuh"
#include "render.cuh"

static void render_row(int y, vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length){
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


// Threads pull rows from a shared atomic counter until none remain.
void cpu_render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length){


    // Shared counter tracking the next unclaimed row. std::atomic ensures threads can increment it concurrently without causing a race condition.
    std::atomic<int> next_row{0};

    // Query the number of logical threads the hardware supports.
    // If a value cannot be determined 4 is used.
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    // Define the work each thread does
    auto worker = [&]() {
        int y;
        while ((y = next_row.fetch_add(1)) < height){
            render_row(y, framebuffer, camera_origin, spheres, num_spheres, light_position, width, height, viewport_width, viewport_height, focal_length);
        }

    };

    // Spawn num_threads worker threads that each run the lambda above.
    // reserve() avoids repeated reallocation as threads are added.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; t++){
        threads.emplace_back(worker);

    }

    // Block until every thread has finished rendering so that the function does not return before rendering is complete.
    for (auto& t: threads)
        t.join();


}