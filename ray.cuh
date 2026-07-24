#pragma once 

#include "vec3.cuh"

struct ray {
    vec3 origin;
    vec3 direction;

    __host__ __device__ ray(const vec3& origin, const vec3& direction) : origin(origin), direction(direction) {}

    __host__ __device__ vec3 at(float t) const {
        return origin + direction * t;
    }
};