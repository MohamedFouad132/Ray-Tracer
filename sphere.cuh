#pragma once

#include "vec3.cuh"

struct sphere {
    vec3 center;
    float radius;
    vec3 color;
    float reflectivity;

    __host__ __device__ sphere(const vec3& center, float radius, const vec3& color, float reflectivity) : center(center), radius(radius), color(color), reflectivity(reflectivity) {}
};