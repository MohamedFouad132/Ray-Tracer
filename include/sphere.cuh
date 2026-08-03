// sphere.cuh
// This file defines the sphere structure found in the ray tracing scene.
// A sphere is defined by its center point, radius, color, and reflectivity.
// A 0.0 reflectivity means the sphere is fully diffuse.
// A 1.0 reflectivity means the sphere is fully reflective.
// Works for both host and device code.

#pragma once
#include "vec3.cuh"

struct sphere {
    vec3 center;
    float radius;
    vec3 color;
    float reflectivity;

    // Default constructor, needed for declaring arrays (e.g. shared memory) before initialization
    __host__ __device__ sphere() : center(0,0,0), radius(0), color(0,0,0), reflectivity(0) {}

    __host__ __device__ sphere(const vec3& center, float radius, const vec3& color, float reflectivity) : center(center), radius(radius), color(color), reflectivity(reflectivity) {}
};