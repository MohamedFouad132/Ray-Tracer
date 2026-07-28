#pragma once

#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include <cfloat>

__host__ __device__ bool hit_sphere(const ray& r, const sphere& s, float& t){
    vec3 oc = r.origin - s.center;
    float a = r.direction.dot(r.direction);
    float b = 2.0f * oc.dot(r.direction);
    float c = oc.dot(oc) - s.radius * s.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
        return false;
        
    float t0 = (-b - sqrtf(discriminant)) / (2.0f * a);
    float t1 = (-b + sqrtf(discriminant)) / (2.0f * a);
    if (t0 > 0.001f) {
        t = t0;
        return true;
    }

    if (t1 > 0.001f) {
        t = t1;
        return true;
    }

    return false;
}

__host__ __device__ int hit_scene(const ray& r, const sphere* spheres, int num_spheres, float& closest_t) {
    closest_t = FLT_MAX;
    int index = -1;
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (hit_sphere(r, spheres[i], t) && t < closest_t) {
            closest_t = t;
            index = i;
        }
    }

    return index;
}


__host__ __device__ bool hit_plane(const ray& r, float plane_y, float& t) {
    if (fabsf(r.direction.y) < 0.0001f) return false;
    t = (plane_y - r.origin.y) / r.direction.y;
    return t > 0.001f;
}



__host__ __device__ vec3 reflect(const vec3& d, const vec3& n) {
    return d - n * 2.0f * d.dot(n);
}