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


__global__ void render(vec3* framebuffer, vec3 camera_origin, sphere* spheres, int num_spheres, vec3 light_position, int width, int height, float viewport_width, float viewport_height, float focal_length) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixel_index = y * width + x;

    if (x >= width || y >= height) return;

    float u = float(x) / (width - 1);
    float v = float(y) / (height - 1);

    float viewport_x = (u-0.5f) * viewport_width;
    float viewport_y = (0.5f-v) * viewport_height;
    vec3 viewport_point(viewport_x, viewport_y, -focal_length);
    vec3 ray_direction = (viewport_point - camera_origin).normalize();
    ray r(camera_origin, ray_direction);

    vec3 final_color(0.0f, 0.0f, 0.0f);
    vec3 color_multiplier(1.0f, 1.0f, 1.0f);
    const int MAX_BOUNCES = 5;
    ray current_ray = r;
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        float hit_t;
        int hit_index = hit_scene(current_ray, spheres, num_spheres, hit_t);
        bool floor_is_closer = hit_floor && (hit_index == -1 || plane_t < hit_t);

        floor hit_plane;
        int hit_index_plane = hit_floor(current_ray, -1.0f, hit_t);
        if (hit_index == -1 && hit_index_plane == -1) {
            float sky_t = 0.5f * (current_ray.direction.y + 1.0f);
            vec3 sky_color = vec3(1.0f, 1.0f, 1.0f) * (1.0f - sky_t) + vec3(0.5f, 0.7f, 1.0f) * sky_t;
            final_color = final_color + sky_color * color_multiplier;
            break;
        }

        if (floor_is_closer) {
            vec3 hit_point = current_ray.at(plane_t);
            int checker = (int(floorf(hit_point.x)) + int(floorf(hit_point.z))) % 2;
            vec3 floor_color = (checker == 0) ? vec3(0.9f,0.9f,0.9f) : vec3(0.1f,0.1f,0.1f);

            vec3 normal(0.0f, 1.0f, 0.0f);
            vec3 light_direction = (light_position - hit_point).normalize();
            float brightness = fmaxf(0.0f, normal.dot(light_direction));

            final_color = final_color + floor_color * brightness * color_multiplier;
            break; 
        }

        vec3 hit_point = current_ray.at(hit_t);
        vec3 normal = (hit_point - spheres[hit_index].center).normalize();
        vec3 light_direction = (light_position - hit_point).normalize();
        float brightness = normal.dot(light_direction);
        brightness = fmaxf(0.0f, brightness);
        ray shadow_ray(hit_point, light_direction);
        float shadow_t;
        int shadow_hit_index = hit_scene(shadow_ray, spheres, num_spheres, shadow_t);
        if (shadow_hit_index != -1 && shadow_t < (light_position - hit_point).length()) {
            brightness *= 0.5f;
        }
        vec3 local_color = spheres[hit_index].color * brightness;
        final_color = final_color + local_color * (1-spheres[hit_index].reflectivity) * color_multiplier;
        if (spheres[hit_index].reflectivity <= 0.0f) break;
        color_multiplier = color_multiplier * spheres[hit_index].reflectivity;
        vec3 reflected_direction = reflect(current_ray.direction, normal);
        current_ray = ray(hit_point, reflected_direction);
    }
    framebuffer[pixel_index] = final_color;

}


int main(){

    sphere h_spheres[3] = {
        sphere(vec3(0.0f, 0.0f, -2.0f), 0.5f, vec3(1.0f, 0.0f, 0.0f), 0.0f),
        sphere(vec3(1.2f, 0.0f, -3.0f), 0.5f, vec3(0.0f, 1.0f, 0.0f), 0.0f),   
        sphere(vec3(-1.2f, 0.3f, -2.5f), 0.4f, vec3(0.0f, 0.0f, 1.0f), 0.6f)   
    };
    int num_spheres = 3;

    sphere* d_spheres;
    cudaMalloc(&d_spheres, num_spheres * sizeof(sphere));
    cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(sphere), cudaMemcpyHostToDevice);

    vec3 light_position(2.0f, 2.0f, 0.0f);

    int num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3 *d_framebuffer;
    cudaMalloc(&d_framebuffer, framebuffer_size);
    vec3 camera_origin(0.0f, 0.0f, 0.0f);

    dim3 blockDim(16, 16);
    dim3 gridDim((IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);
    render<<<gridDim, blockDim>>>(d_framebuffer, camera_origin, d_spheres, num_spheres, light_position, IMAGE_WIDTH, IMAGE_HEIGHT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_LENGTH);
    cudaDeviceSynchronize();
    
    vec3* h_framebuffer = (vec3*)malloc(framebuffer_size);
    cudaMemcpy(h_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost);
    FILE* f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);

    for (int i = 0; i < num_pixels; i++) {
        vec3 color = h_framebuffer[i];
        int r = int(color.x * 255);
        int g = int(color.y * 255);
        int b = int(color.z * 255);
        fprintf(f, "%d %d %d\n", r, g, b);
    }

    fclose(f);

    free(h_framebuffer);
    cudaFree(d_framebuffer);
    cudaFree(d_spheres);
    return 0;
}