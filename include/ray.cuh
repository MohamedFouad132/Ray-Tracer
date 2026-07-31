// ray.cuh
// This file defines the ray structure used in ray tracing.
// A ray is defined by an origin point and a normalized direction vector
// Works for both host and device code.

#pragma once 
#include "vec3.cuh"

struct ray {
    vec3 origin;
    vec3 direction;

    __host__ __device__ ray(const vec3& origin, const vec3& direction) : origin(origin), direction(direction) {}

    // Finds a point along the ray at a given distance t from the ray's origin.
    __host__ __device__ vec3 at(float t) const {
        return origin + direction * t;
    }
};