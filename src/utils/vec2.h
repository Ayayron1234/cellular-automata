#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using Float = double;

struct vec2 {
	Float x;
	Float y;

	__host__ __device__ vec2(Float x0 = 0, Float y0 = 0) { x = x0; y = y0; }
	__host__ __device__ vec2 operator*(Float a) const { return vec2(x * a, y * a); }
	__host__ __device__ vec2 operator/(Float a) const { return vec2(x / a, y / a); }
	__host__ __device__ vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
	__host__ __device__ vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
	__host__ __device__ vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }
	__host__ __device__ vec2 operator-() const { return vec2(-x, -y); }
};

__host__ __device__ inline Float dot(const vec2& v1, const vec2& v2) {
	return (v1.x * v2.x + v1.y * v2.y);
}

__host__ __device__ inline Float length(const vec2& v) { return sqrtf(dot(v, v)); }

__host__ __device__ inline vec2 normalize(const vec2& v) { return v * (1 / length(v)); }

__host__ __device__ inline vec2 operator*(Float a, const vec2& v) { return vec2(v.x * a, v.y * a); }
