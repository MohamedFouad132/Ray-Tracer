// intersect.cuh
// This file defines functions to determine intersections between rays and objects in the scene.
// For formula derivations, see docs/derivations.md
// Works for both host and device code.

#pragma once
#include <cfloat>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

// Determines if a ray intersects with a sphere and finds the intersection distance t if it does.
// The function uses the quadratic formula to solve for t, where t is the distance along the ray to the intersection point.
// For the full derivation, see docs/derivations.md#sphere-intersection
__host__ __device__ inline bool hit_sphere(const ray& r, const sphere& s, float& t){

    vec3 oc = r.origin - s.center;
    // coefficents for the quadratic equation at^2 + bt + c = 0
    float a = r.direction.dot(r.direction);
    float b = 2.0f * oc.dot(r.direction);
    float c = oc.dot(oc) - s.radius * s.radius;

    // Calculate the discriminant to determine if there are real roots (intersections)
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
        return false;
    
    // Calculate the two possible intersection distances (t0 and t1)
    float t0 = (-b - sqrtf(discriminant)) / (2.0f * a);
    float t1 = (-b + sqrtf(discriminant)) / (2.0f * a);

    // The condition t > 0.001f is used to avoid self-intersection issues due to floating-point precision and to ensure we only consider intersections in the positive direction of the ray.
    // We want the closest positive intersection distance, so we check t0 first, then t1
    if (t0 > 0.001f) {
        t = t0;
        return true;
    }

    // If t0 is not valid, check t1
    if (t1 > 0.001f) {
        t = t1;
        return true;
    }

    // If neither t0 nor t1 is valid, there is no intersection in the positive direction of the ray
    return false;
}

// Determines if a ray intersects with a horizontal plane at a given z-coordinate and finds the intersection distance t if it does.
// For the derivation, see docs/derivations.md#plane-intersection
__host__ __device__ inline bool hit_plane(const ray& r, float plane_z, float& t) {
    
    // if ray is nearly parallel to the plane then we consider it as not intersecting to avoid numerical instability
    // This is because it would need to travel an unrealistic distance to intersect the plane which is not practical
    if (fabsf(r.direction.z) < 0.0001f) return false;
    // origin + direction * t = plane_z
    t = (plane_z - r.origin.z) / r.direction.z;
    // Check that t is positive and greater than a small threshold to avoid self-intersection issues
    return t > 0.001f;
}

// Determines the closest intersection of a ray with a scene of spheres.
// It iterates through all spheres checking for intersections and keeping track of the closest one found.
// Returns the index of the closest intersected sphere and updates closest_t with the distance to that intersection.
__host__ __device__ inline int hit_scene(const ray& r, const sphere* spheres, int num_spheres, float& closest_t) {
    // Initialize closest_t to a large value to ensure any valid intersection will be closer
    closest_t = FLT_MAX;
    int index = -1;
    for (int i = 0; i < num_spheres; i++) {
        float t;
        // Check and update closest intersection if t is less than the current closest_t
        if (hit_sphere(r, spheres[i], t) && t < closest_t) {
            closest_t = t;
            index = i;
        }
    }

    // Return the index of the closest intersected sphere or -1 if no intersection was found
    return index;
}

// Determines if a ray intersects with any sphere in the scene within a maximum distance max_t.
// This function is used for shadows to exit early if any object blocks the light source.
__host__ __device__ inline bool hit_anything(const ray& r, const sphere* spheres, int num_spheres, float max_t) {

    for (int i = 0; i < num_spheres; i++) {
        float t;
        // If the ray intersects with a sphere and the intersection distance t is less than max_t then return true indicating an intersection.
        if (hit_sphere(r, spheres[i], t) && t < max_t) {
            return true;
        }
    }

    return false;
}


// Reflects a vector d around a normal n using the reflection formula
// For the derivation of this formula, see docs/derivations.md#reflection
__host__ __device__ inline vec3 reflect(const vec3& d, const vec3& n) {
    return d - n * 2.0f * d.dot(n);
}