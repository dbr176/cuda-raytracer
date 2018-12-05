
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "math_helper.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define TILE_WIDTH 8
#define EPS 0.0000001f

using namespace cv;
using namespace std;

struct Ray {
	float3 direction;
	float3 origin;
};

struct ColorRay {
	Ray ray;
	float4 color;
};

struct Collision {
	float3 point;
	float3 normal;
	Ray ray;
};

struct Camera {
	float3 position;
	float3 fwd;
	float3 right;
	float3 up;

	float w;
	float h;
};

struct Light {
	float3 pos;
	float4 color;
};

struct Tri {
	float3 a, b, c;
};

struct Mesh {
	std::vector<float3> points;
	std::vector<int3> faces;
};

std::vector<Tri> meshToTriangles(Mesh mesh) {
	std::vector<Tri> res = std::vector<Tri>();

	for (int i = 0; i < mesh.faces.size(); i++) {
		Tri tri;
		tri.a = mesh.points[mesh.faces[i].x];
		tri.b = mesh.points[mesh.faces[i].y];
		tri.c = mesh.points[mesh.faces[i].z];
		res.push_back(tri);
	}

	return res;
}

auto g_right = make_float3(1, 0, 0);
auto g_fwd = make_float3(0, 0, 1);
auto g_up = make_float3(0, 1, 0);

void addTri(std::vector<int3>& faces, int a, int b, int c) {
	faces.push_back(make_int3(a, b, c));
}

void addQuad(std::vector<int3>& faces, int a, int b, int c, int d) {
	addTri(faces, d, a, b);
	addTri(faces, b, c, d);
}

Mesh cube() {
	Mesh res;

	std::vector<float3> points = std::vector<float3>();
	std::vector<int3> faces = std::vector<int3>();

	auto zero = make_float3(0, 0, 0);
	points.push_back(zero);
	points.push_back(g_up);
	points.push_back(g_up + g_right);
	points.push_back(g_up + g_fwd);
	points.push_back(g_up + g_fwd + g_right);
	points.push_back(g_right);
	points.push_back(g_right + g_fwd);
	points.push_back(g_fwd);

	addQuad(faces, 0, 5, 2, 1);
	addQuad(faces, 0, 1, 3, 7);
	addQuad(faces, 0, 7, 6, 5);
	addQuad(faces, 5, 6, 4, 2);
	addQuad(faces, 7, 3, 4, 6);
	addQuad(faces, 4, 3, 1, 2);

	res.faces = faces;
	res.points = points;

	return res;
}

void translate(Mesh &mesh, float3 dir) {
	for (int i = 0; i < mesh.points.size(); i++)
		mesh.points[i] += dir;
}

void translate(Mesh &mesh, float x, float y, float z) {
	translate(mesh, make_float3(x, y, z));
}

float recenterX(Camera cam, float x) {
	return (x - cam.w / 2.0f) / (2.0f * cam.w);
}

float recenterY(Camera cam, float y) {
	return -(y - cam.h / 2.0f) / (2.0f * cam.h);
}

Ray getRay(Camera cam, float x, float y) {
	auto nx = (x / cam.w - 0.5f) * 2.0f * cam.right;
	auto ny = -(y / cam.h - 0.5f) * 2.0f * cam.up;

	Ray ray;
	ray.direction = normalize((nx + ny + cam.fwd) - cam.position);
	ray.origin = cam.position;

	return ray;
}

__device__ Ray fromTo(float3 from, float3 to) {
	Ray ray;
	ray.origin = from;
	ray.direction = normalize(to - from);

	return ray;
}

__device__ float3 triNormal(Tri tri) {
	auto e1 = tri.b - tri.a;
	auto e2 = tri.c - tri.a;

	return normalize(e1 * e2);
}

__device__ __host__ float3 barTest(Ray ray, float3 v1, float3 v2, float3 v3)
{
	float epsilon = 0.0000001f;

	float3 d = ray.direction;
	float3 e1 = v2 - v1;
	float3 e2 = v3 - v1;

	float3 p = cross(d, e2);
	float det = dot(e1, p);

	if (fabs(det) < epsilon) return make_float3(0, 0, 0);

	auto invDet = 1.0f / det;
	auto t = ray.origin - v1;
	auto u = dot(t, p) * invDet;

	if (u < 0.0f || u > 1.0f) return make_float3(0, 0, 0);

	auto q = cross(t, e1);
	auto v = dot(d, q) * invDet;

	if (v < 0.0f || u + v > 1.0f) return make_float3(0, 0, 0);

	auto nt = dot(e2, q) * invDet;

	if (nt > epsilon)
		return nt * ray.direction + ray.origin;

	return make_float3(0, 0, 0);
}

__device__ __host__ float3 barTestTri(Ray ray, Tri tri) {
	return barTest(ray, tri.a, tri.b, tri.c);
}

__device__ float shadowRay(
	float3 p,
	Tri* triangles,
	int triCount, float3* lights, int lightCount) {
	float intencity = 0;
	for (int lightIdx = 0; lightIdx < lightCount; lightIdx++) {
		bool seeLight = true;
		Ray ray = fromTo(p, lights[lightIdx]);

		for (int triIdx = 0; triIdx < triCount && seeLight; triIdx++) {
			Tri tri = triangles[triIdx];

			auto test = barTestTri(ray, tri);
			if (test.x == 0 && test.y == 0 && test.z == 0) {
				intencity += 0.1f;
				seeLight = false;
			}
		}
		if (seeLight)
			intencity += 0.1f;
	}
	return intencity;
}

#define DEPTH 4

__global__ void testHits(
	Ray* rays,
	Tri* tris, int triCount,
	int width, int height,
	float4* result
) {
	int rayIdx = threadIdx.x + blockIdx.x*blockDim.x;

	// Light position
	float3 lightPos = make_float3(-1.0f, 0, 10);

	if (rayIdx < width * height) {
		Ray ray = rays[rayIdx];

		// Ambient color
		result[rayIdx] = make_float4(0.00f, 0.00f, 0.00f, 1);

		float3 hitPoints[DEPTH];
		float3 normals[DEPTH];
		float distances[DEPTH];
		float4 materialColor[DEPTH];
		float4 shadowColors[DEPTH];

		int hits = 0;
		bool hit = false;

		for (int depth = 0; depth < DEPTH; ++depth) {
			hit = false;
			distances[depth] = 100000000000.0f;

			for (int triIdx = 0; triIdx < triCount; ++triIdx) {
				auto tri = tris[triIdx];
				auto hitPoint = barTestTri(ray, tri);

				if (fabs(hitPoint.x) > EPS || fabs(hitPoint.y) > EPS || fabs(hitPoint.z) > EPS) {
					auto e1 = tri.c - tri.a;
					auto e2 = tri.c - tri.b;

					auto normal = normalize(cross(e1, e2));
					auto dist = dot(ray.origin - hitPoint, ray.origin - hitPoint);
					
					// hit point could be behind the triangle
					// so I moved it 
					hitPoint = hitPoint - 0.0001f * normal;

					if (distances[depth] > dist) {
						hit = true;
						hitPoints[depth] = hitPoint;
						distances[depth] = dist;
						normals[depth] = normal;
						// material color
						materialColor[depth] = make_float4(0.5f, 0.5f, 0.5f, 1);
					}
				}
			}

			if (!hit) break;
			hits++;
			ray.origin = hitPoints[depth];
			ray.direction = reflect(ray.direction, normals[depth]);
		}

		for (int hitIdx = hits - 1; hitIdx >= 0; --hitIdx) {
			auto toLight = lightPos - hitPoints[hitIdx];
			Ray shadowRay;
			shadowRay.direction = toLight;
			shadowRay.origin = hitPoints[hitIdx];

			hit = false;

			for (int triIdx = 0; triIdx < triCount; ++triIdx) {
				auto tri = tris[triIdx];
				auto hitPoint = barTestTri(shadowRay, tri);

				if (fabs(hitPoint.x) > EPS || fabs(hitPoint.y) > EPS || fabs(hitPoint.z) > EPS) {
					auto dist = dot(shadowRay.origin - hitPoint, shadowRay.origin - hitPoint);
					hit = true;
					break;
				}
			}

			if (!hit)
			{
				// Calculate light color and intensity
				auto dist = 3 / sqrt(dot(lightPos - hitPoints[hitIdx], lightPos - hitPoints[hitIdx]));
				shadowColors[hitIdx] = (make_float4(dist, dist, dist, 1) + make_float4(0.1f, 0.1f, 0.1f, 1)) / (hitIdx + 1);
			}
			else {
				// Ambient light color
				shadowColors[hitIdx] = make_float4(0.00f, 0.02f, 0.0f, 1);
			}
		}

		for (int hitIdx = hits - 1; hitIdx >= 0; --hitIdx) {
			result[rayIdx] += materialColor[hitIdx] * shadowColors[hitIdx];
		}
	}
}

void testHitsDevice(
	Ray* rays,
	Tri* tris, int triCount,
	int width, int height,
	float4* result
) {
	for (int rayIdx = 0; rayIdx < width * height; rayIdx++) {
		float3 lightPos = make_float3(0, 10, 10);
		Ray ray = rays[rayIdx];

		result[rayIdx] = make_float4(0.00f, 0.00f, 0.00f, 1);
		int hits = 0;

		float3 hitPoints[DEPTH];
		float3 normals[DEPTH];
		float distances[DEPTH];
		float4 materialColor[DEPTH];
		float4 shadowColors[DEPTH];

		bool hit = false;

		for (int depth = 0; depth < DEPTH; ++depth) {


			for (int triIdx = 0; triIdx < triCount; ++triIdx) {
				auto tri = tris[triIdx];
				auto hitPoint = barTestTri(ray, tri);

				if (hitPoint.x != 0 || hitPoint.y != 0 || hitPoint.z != 0) {
					auto e1 = tri.c - tri.a;
					auto e2 = tri.c - tri.b;

					auto normal = normalize(cross(e1, e2));
					auto dist = dot(ray.origin, hitPoint);
					hitPoint = hitPoint + 0.00001f * normal;

					hit = true;

					if (distances[depth] < dist) {
						hitPoints[depth] = hitPoint;
						distances[depth] = dist;
						normals[depth] = normal;
						materialColor[depth] = make_float4(0.1f, 0.0f, 0.0f, 1);
					}
				}
			}

			if (hit) hits++;

			if (!hit) break;

			if (hits == 2)
				result[rayIdx] = make_float4(1.1f, 1.1f, 1.1f, 1);

			ray.origin = hitPoints[depth];
			ray.direction = reflect(ray.direction, normals[depth]);
		}

		for (int hitIdx = hits - 1; hitIdx >= 0; --hitIdx) {
			auto toLight = lightPos - hitPoints[hitIdx];
			Ray shadowRay;
			shadowRay.direction = toLight;
			shadowRay.origin = hitPoints[hitIdx];

			hit = false;

			for (int triIdx = 0; triIdx < triCount; ++triIdx) {
				auto tri = tris[triIdx];
				auto hitPoint = barTestTri(ray, tri);
				if (hitPoint.x != 0 || hitPoint.y != 0 || hitPoint.z != 0) {
					hit = true;
				}
			}

			if (!hit)
			{
				shadowColors[hitIdx] = make_float4(0.1f, 0.1f, 0.1f, 1);
			}
		}

		for (int hitIdx = hits - 1; hitIdx >= 0; --hitIdx) {
			//result[rayIdx] = 10 * shadowColors[hitIdx];
			//result[rayIdx] = make_float4(1.1f, 1.1f, 1.1f, 1);
		}

		//result[rayIdx] /= DEPTH;
	}
}

void printVec(float3 vec) {
	std::cout << "(" << vec.x << " " << vec.y << " " << vec.z << ")" << std::endl;
}

std::vector<Tri> createScene() {
	auto tris = std::vector<Tri>();
	auto cubes = std::vector<Mesh>();
	for (int i = 0; i < 3; i++)
		cubes.push_back(cube());

	// 0, -x, 0 -> сдвиг вправо
	// -x, 0, 0 -> свиг вверх

	translate(cubes[0], -0.0f, 0.0f, 5.0f);
	translate(cubes[1], -0.0f, -2.0f, 5.0f);
	translate(cubes[2], -0.0f, 2.0f, 5.0f);

	for (int i = 0; i < cubes.size(); i++) {
		auto ts = meshToTriangles(cubes[i]);

		for (int t = 0; t < ts.size(); t++)
			tris.push_back(ts[t]);
	}

	return tris;
}

void printCam(Camera cam) {
	std::cout << "height: " << cam.h << std::endl;
	std::cout << "width: " << cam.w << std::endl;

	std::cout << "fwd: ";
	printVec(cam.fwd);

	std::cout << "right: ";
	printVec(cam.right);

	std::cout << "up: ";
	printVec(cam.up);

	std::cout << "pos: ";
	printVec(cam.position);

	auto r = getRay(cam, 256, 100);

	std::cout << "origin: ";
	printVec(r.origin);

	std::cout << "direction: ";
	printVec(r.direction);
}

std::vector<Ray> prepareRays(int width, int height) {
	Camera cam;
	cam.position = make_float3(0, 0, 0);

	cam.right = g_right;
	cam.fwd = g_fwd;
	cam.up = g_up;

	cam.w = width;
	cam.h = height;

	printCam(cam);

	std::vector<Ray> hostRay = std::vector<Ray>();

	for (int w = 0; w < width; w++) {
		for (int h = 0; h < height; h++) {
			int idx = w + h * width;
			Ray ray = getRay(cam, w, h);

			hostRay.push_back(ray);
		}
	}

	return hostRay;
}


int main()
{
	int width = 512;
	int height = 512;

	int size = width * height;

	auto scene = createScene();
	auto rays = prepareRays(width, height);

	Tri floor;
	floor.a = make_float3(1, -100.0f, 0.0f);
	floor.b = make_float3(1, 100.0f, 0.0f);
	floor.c = make_float3(1, -0.5f, 1000.0f);
	scene.push_back(floor);

	cudaSetDevice(0);

	Tri *deviceScene = 0;
	Ray *deviceRays = 0;
	float4 *deviceColors = 0;

	Mat image(width, height, CV_32FC4, Scalar(0, 0, 0, 0));
	auto res = (float4*)image.data;

	cudaMalloc(&deviceScene, scene.size() * sizeof(Tri));
	cudaMalloc(&deviceRays, rays.size() * sizeof(Ray));
	cudaMalloc(&deviceColors, size * sizeof(float4));

	cudaMemcpy((void*)deviceScene, (void*)scene.data(), scene.size() * sizeof(Tri), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)deviceRays, (void*)rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice);

	testHits <<<(size + TILE_WIDTH - 1) / TILE_WIDTH, TILE_WIDTH >>> (deviceRays, deviceScene, scene.size(), width, height, deviceColors);

	//testHitsDevice(rays.data(), scene.data(), scene.size(), width, height, res);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "testHits launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching testHits!\n", cudaStatus);
	}

	cudaStatus = cudaMemcpy((void*)res, (void*)deviceColors, size * sizeof(float4), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	imshow("Display window", image);
	waitKey(0);

	cudaFree(deviceScene);
	cudaFree(deviceRays);
	cudaFree(deviceColors);

	return 0;
}