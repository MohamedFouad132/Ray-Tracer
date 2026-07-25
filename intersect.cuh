#pragma once

#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

__host__ __device__ bool hit_sphere(const ray& r, const sphere& s, float& t){
    vec3 oc = r.origin - s.center;
    float a = r.direction.dot(r.direction);
    float b = 2.0f * oc.dot(r.direction);
    float c = oc.dot(oc) - s.radius * s.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
        return false;
        
    t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return true;
}