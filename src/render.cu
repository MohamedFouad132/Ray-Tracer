#include <cstdio>
#include <curand_kernel.h>
#include <random>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "intersect.cuh"
#include "render.cuh"

const int SAMPLES_PER_PIXEL = 4;

// Traces a single ray through the scene, calculating color contributions from direct lighting, shadows, and reflections.
// It returns the final color for the ray.
__host__ __device__ vec3 trace_ray(ray r, sphere* spheres, int num_spheres, vec3 light_position){

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

            // Shadow ray to determine if the hit point is in shadow
            ray shadow_ray(hit_point, light_direction);
            float shadow_t;
            int shadow_hit_index = hit_scene(shadow_ray, spheres, num_spheres, shadow_t);
            if (shadow_hit_index != -1 && shadow_t < (light_position - hit_point).length()) {
                brightness *= 0.5f;
            }

            // Scale the floor color by brightness and the current color multiplier to account for any previous reflections
            final_color = final_color + floor_color * brightness * color_multiplier;
            break;

        }


        // If we hit a sphere we calculate the color contribution from that sphere
        vec3 hit_point = current_ray.at(hit_t);

        // Brightness is calculated based on the angle between the light direction and the normal at the hit point.
        vec3 normal = (hit_point - spheres[hit_index].center).normalize();
        vec3 light_direction = (light_position - hit_point).normalize();
        float brightness = normal.dot(light_direction);
        brightness = fmaxf(0.0f, brightness);


        // Shadow ray to determine if the hit point is in shadow
        ray shadow_ray(hit_point, light_direction);
        float shadow_t;
        int shadow_hit_index = hit_scene(shadow_ray, spheres, num_spheres, shadow_t);

        // If the shadow ray hits another sphere before reaching the light source brightness is halved.
        if (shadow_hit_index != -1 && shadow_t < (light_position - hit_point).length()) {
            brightness *= 0.5f;
        }

        // Calculate the local color based on the sphere's color and the calculated brightness.
        vec3 local_color = spheres[hit_index].color * brightness;
        // Update the final color by adding the local color contribution scaled by the sphere's reflectivity and the current color multiplier.
        final_color = final_color + local_color * (1-spheres[hit_index].reflectivity) * color_multiplier;

        // If the sphere is not reflective we can stop tracing further rays.
        if (spheres[hit_index].reflectivity <= 0.0f) break;

        // Update the color multiplier for the next bounce based on the sphere's reflectivity and calculate the reflected ray direction.
        color_multiplier = color_multiplier * spheres[hit_index].reflectivity;
        vec3 reflected_direction = reflect(current_ray.direction, normal);
        current_ray = ray(hit_point, reflected_direction);
    }

    return final_color; // Return the accumulated color after all bounces and contributions have been calculated.
}


__global__ void gpu_render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length) {

    // Each thread computes the final color for exactly one pixel.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_index = y * width + x;

    // Guard against threads outside the image
    if (x >= width || y >= height) return;


    // Initialize a random number generator for anti-aliasing.
    // The curandState is used to generate random numbers for jittering the ray direction within each pixel.
    curandState rand_state;
    curand_init(pixel_index, 0, 0, &rand_state);

    // Initialize vector to accumulate color samples for anti-aliasing
    vec3 pixel_color(0.0f, 0.0f, 0.0f);

    // For each sample, jitter the ray direction slightly to achieve anti-aliasing.
    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {

        // Generate random offsets for anti-aliasing within the pixel
        float jitter_x = curand_uniform(&rand_state) - 0.5f;
        float jitter_y = curand_uniform(&rand_state) - 0.5f;

        // Compute fraction of the pixel's position in the viewport adjusted by the jitter
        float u = (float(x) + jitter_x) / (width - 1);
        float v = (float(y) + jitter_y) / (height - 1);
        
        float viewport_x = (u-0.5f) * viewport_width;
        float viewport_z = (0.5f-v) * viewport_height + camera_origin.z; // Adjusted to move the viewport in front of the camera

        // Compute ray direction and create a ray from the camera origin through the computed viewport point
        vec3 viewport_point(viewport_x, focal_length, viewport_z);
        vec3 ray_direction = (viewport_point - camera_origin).normalize();
        ray r(camera_origin, ray_direction);

        // Trace the ray and accumulate the color contribution from this sample
        pixel_color = pixel_color + trace_ray(r, spheres, num_spheres, light_position);
    }

    // Average the accumulated color samples to get the final pixel color and store it in the framebuffer
    pixel_color = pixel_color * (1.0f / SAMPLES_PER_PIXEL);
    framebuffer[pixel_index] = pixel_color;
}



void cpu_render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length){
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            int pixel_index = y * width + x;
            std::mt19937 rng(pixel_index);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            vec3 pixel_color(0.0f, 0.0f, 0.0f);
            for (int s = 0; s < SAMPLES_PER_PIXEL; s++){
                float jitter_x = dist(rng);
                float jitter_y = dist(rng);

                float u = (float(x) + jitter_x) / (width - 1);
                float v = (float(y) + jitter_y) / (height - 1);

                float viewport_x = (u-0.5f) * viewport_width;
                float viewport_z = (0.5f-v) * viewport_height + camera_origin.z; // Adjusted to move the viewport in front of the camera

                // Compute ray direction and create a ray from the camera origin through the computed viewport point
                vec3 viewport_point(viewport_x, focal_length, viewport_z);
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