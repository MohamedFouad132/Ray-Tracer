// vec3.cuh
// This file defines a simple 3D vector structure with basic operations.
// Operations include: add, subtract, scale, dot product, normalize, and length calculation.
// Used to represent points, directions, and colors in 3D space.
// Works for both host and device code.

#pragma once

struct vec3 {
    float x, y, z;
    
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}


    // Overload operators for vector arithmetic
    // Multiplication is defined for both scalar and component-wise multiplication.
    __host__ __device__ vec3 operator+(const vec3& other) const {
        return vec3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ vec3 operator-(const vec3& other) const {
        return vec3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ vec3 operator*(float scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ vec3 operator*(const vec3& other) const {
        return vec3(x * other.x, y * other.y, z * other.z);
    }
    
    // Dot product between two vectors
    __host__ __device__ float dot(const vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Length of the vector (sqrt of a vector's dot product with itself)
    __host__ __device__ float length() const {
        return sqrtf(dot(*this));
    }


    // Normalize the vector to have a length of 1
    // Dividing each component by the vector's length scales it to unit length without changing its direction.
    __host__ __device__ vec3 normalize() const {
        float len = length();
        return vec3(x / len, y / len, z / len);
    }

};
