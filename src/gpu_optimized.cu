// gpu_optimized.cu
// This file contains the optimized GPU implementation of the ray tracing algorithm.

#include <cstdio>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "intersect.cuh"
#include "render.cuh"

// Generates a pseudo-random float in the range [0, 1) using a simple linear congruential generator (LCG).
__device__ float random_float(unsigned int& seed) {
    seed = seed * 1664525u + 1013904223u;
    return float(seed & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Optimized version of trace_ray
// Implements early exit conditions for shadow rays to avoid unnecessary calculations
// Caches the hit sphere data in a local variable to avoid repeated memory accesses
__host__ __device__ vec3 trace_ray_optimized(ray r, sphere* spheres, int num_spheres, vec3 light_position){

    const int MAX_BOUNCES = 5;

    // Initialize the final color to black (no light contribution)
    vec3 final_color(0.0f, 0.0f, 0.0f);
    // tracks how much each subsquent reflection contributes to the final color (shrinks with each reflection)
    vec3 color_multiplier(1.0f, 1.0f, 1.0f);

    ray current_ray = r;


    // Loop through a maximum number of bounces to simulate reflections
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        
        // Check for intersection with scene
        float hit_t;
        int hit_index = hit_scene(current_ray, spheres, num_spheres, hit_t);
        
        // Check for intersection with the floor plane at z = 0.0
        float plane_t;
        bool hit_floor = hit_plane(current_ray, 0.0f, plane_t);


        // If it hits both check which one it hits first (i.e smallest t value)
        bool floor_is_closer = hit_floor && (hit_index == -1 || plane_t < hit_t);

        // If it hits neither, we can assume the ray goes off into the sky and we can use a gradient for the sky color
        if (hit_index == -1 && !hit_floor) {
            // sky_t is calculated based on the z-component of the ray's direction, which ranges from -1 to 1.
            // This value is then mapped to a range of 0 to 1 to then be used to mix the colors white and light blue.
            float sky_t = 0.5f * (current_ray.direction.z + 1.0f);
            vec3 sky_color = vec3(1.0f, 1.0f, 1.0f) * (1.0f - sky_t) + vec3(0.5f, 0.7f, 1.0f) * sky_t;
            // Scale by color_multiplier since this sky hit might be several bounces deep
            final_color = final_color + sky_color * color_multiplier;
            break;
        }

        // If the floor is closer than any sphere we handle the floor intersection
        if (floor_is_closer) {
            // Find hit point on the floor and determine checkerboard pattern color based on x and z coordinates
            vec3 hit_point = current_ray.at(plane_t);
            // The checkerboard pattern is determined by the sum of the floored x and y coordinates of the hit point.
            int checker = ((int(floorf(hit_point.x)) + int(floorf(hit_point.y))) % 2 + 2) % 2;
            // Assign light gray color for even squares and dark gray for odd squares.
            vec3 floor_color = (checker == 0) ? vec3(0.9f,0.9f,0.9f) : vec3(0.1f,0.1f,0.1f);

            vec3 normal(0.0f, 0.0f, 1.0f);
            vec3 light_direction = (light_position - hit_point).normalize();
            // Calculate brightness based on the angle between the light direction and the floor's normal vector.
            float brightness = fmaxf(0.0f, normal.dot(light_direction));

            // Uses hit_anything to allow for early exit if the shadow ray hits any sphere before reaching the light source
            ray shadow_ray(hit_point, light_direction);
            if (hit_anything(shadow_ray, spheres, num_spheres, (light_position - hit_point).length())) {
                brightness *= 0.5f;
            }

            // Scale the floor color by brightness and the current color multiplier to account for any previous reflections
            final_color = final_color + floor_color * brightness * color_multiplier;
            break;

        }


        // If we hit a sphere we calculate the color contribution from that sphere
        vec3 hit_point = current_ray.at(hit_t);

        sphere hit_sphere_data = spheres[hit_index]; // Store the hit sphere's data in a local variable to avoid repeated memory accesses.

        // Brightness is calculated based on the angle between the light direction and the normal at the hit point.
        vec3 normal = (hit_point - hit_sphere_data.center).normalize();
        vec3 light_direction = (light_position - hit_point).normalize();
        float brightness = normal.dot(light_direction);
        brightness = fmaxf(0.0f, brightness);


        // Uses hit_anything to allow for early exit if the shadow ray hits any sphere before reaching the light source
        ray shadow_ray(hit_point, light_direction);
        if (hit_anything(shadow_ray, spheres, num_spheres, (light_position - hit_point).length())) {
            brightness *= 0.5f;
        }

        // Calculate the local color based on the sphere's color and the calculated brightness.
        vec3 local_color = hit_sphere_data.color * brightness;
        // Update the final color by adding the local color contribution scaled by the sphere's reflectivity and the current color multiplier.
        final_color = final_color + local_color * (1-hit_sphere_data.reflectivity) * color_multiplier;

        // If the sphere is not reflective we can stop tracing further rays.
        if (hit_sphere_data.reflectivity <= 0.0f) break;

        // Update the color multiplier for the next bounce based on the sphere's reflectivity and calculate the reflected ray direction.
        color_multiplier = color_multiplier * hit_sphere_data.reflectivity;
        vec3 reflected_direction = reflect(current_ray.direction, normal);
        current_ray = ray(hit_point, reflected_direction);
    }

    return final_color; // Return the accumulated color after all bounces and contributions have been calculated.
}


// Optimized version of gpu_render
// Uses a LCG for random number generation instead of curand to allow for more occupancy
// Uses shared memory for sphere data to reduce global memory accesses
__global__ void gpu_render_optimized(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length) {

    // Load sphere data into shared memory to reduce  memory accesses
    __shared__ sphere shared_spheres[4];
    int local_index = threadIdx.y * blockDim.x + threadIdx.x;
    if (local_index < num_spheres) {
        shared_spheres[local_index] = spheres[local_index];
    }
    __syncthreads();
    

    // Each thread computes the final color for exactly one pixel.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_index = y * width + x;

    // Guard against threads outside the image
    if (x >= width || y >= height) return;

    // Seed the LCG with a unique value for each pixel to ensure different random values for each pixel
    unsigned int seed = pixel_index * 3242174893u + 123456789u;

    // Initialize vector to accumulate color samples for anti-aliasing
    vec3 pixel_color(0.0f, 0.0f, 0.0f);

    // For each sample, jitter the ray direction slightly to achieve anti-aliasing.
    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {

        // Generate random offsets for anti-aliasing within the pixel
        float jitter_x = random_float(seed) - 0.5f;
        float jitter_y = random_float(seed) - 0.5f;

        // Compute fraction of the pixel's position in the viewport adjusted by the jitter
        float u = (float(x) + jitter_x) / (width - 1);
        float v = (float(y) + jitter_y) / (height - 1);
        
        float viewport_x = (u-0.5f) * viewport_width;
        float viewport_z = (0.5f-v) * viewport_height + camera_origin.z; // Adjusted to move the viewport in front of the camera

        // Compute ray direction and create a ray from the camera origin through the computed viewport point
        vec3 viewport_point(viewport_x, camera_origin.y + focal_length, viewport_z);
        vec3 ray_direction = (viewport_point - camera_origin).normalize();
        ray r(camera_origin, ray_direction);

        // Trace the ray and accumulate the color contribution from this sample using the optimized trace_ray function
        pixel_color = pixel_color + trace_ray_optimized(r, shared_spheres, num_spheres, light_position);
    }

    // Average the accumulated color samples to get the final pixel color and store it in the framebuffer
    pixel_color = pixel_color * (1.0f / SAMPLES_PER_PIXEL);
    framebuffer[pixel_index] = pixel_color;
}
