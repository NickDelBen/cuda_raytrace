
#include "raytracer.h"

void Raytracer(color_t * d_f, line_t * d_r, world_t * w, int size,
	int blocks, int threads, int max_reflections)
{
	int b_work = size / blocks,
		t_work = b_work / threads,
		r_size = sizeof(line_t) * b_work,
		f_size = sizeof(color_t) * b_work;

	// Copy animated world to device.
	world_t * d_w = World_toDevice(w);

	// Initialize frame
	Frame_init<<<blocks, threads, f_size>>>(d_f, size);

	line_t * rays;
	cudaMalloc(&rays, sizeof(line_t) * size);
	cudaMemcpy(rays, d_r, sizeof(line_t) * size, cudaMemcpyDeviceToDevice);

	// Traces rays bounces.
	for (int i = 0; i < max_reflections; ++i) {
		Raytracer_trace<<<blocks, threads, r_size + f_size>>>(rays, d_f, d_w,
			size, b_work, t_work);
	}

	// Frees world from device memory.
	World_freeDevice(d_w);
}

__global__ void Raytracer_trace (line_t * d_r, color_t * d_f, world_t * d_w,
	int size, int b_work, int t_work)
{
	int b_offset 	= blockIdx.x * b_work,
		t_offset 	= threadIdx.x * t_work,
		offset 		= b_offset + t_offset;

	extern __shared__ float smem[];

	// Assign shared memory locations to the rays and frame arrays.
	line_t 	* rays 	= (line_t*)smem;
	color_t * frame = (color_t*)&rays[b_work],
			result;

	// Copy from global memory to shared memory.
	memcpy(&rays[t_offset], &d_r[offset], sizeof(line_t) * t_work);
	memcpy(&frame[t_offset], &d_f[offset], sizeof(color_t) * t_work);

	// Process all the pixels assigned to this thread
	for (int i = t_offset; i < t_offset + t_work; ++i) {
		Raytracer_calculatePixelColor(&result, d_w, &rays[i]);
		COLOR_ADD(&frame[i], &frame[i], &result);
	}

	// Copy the results of the trace on the frame tile to the global memory.
	memcpy(&d_r[offset], &rays[t_offset], sizeof(line_t) * t_work);
	memcpy(&d_f[offset], &frame[t_offset], sizeof(color_t) * t_work);
}

__device__ void Raytracer_calculatePixelColor (color_t * color, world_t  * d_w,
	line_t * ray)
{
    color_t bg = d_w->bg;

    //sets the pixel color to the default background color
    color->r = bg.r;
    color->b = bg.b;
    color->g = bg.g;

    float distance = -1, temp;
    object_t * object = NULL;

    object_t * objects = d_w->objects;
    for (int i = 0; i < d_w->n_objects; ++i) {
    	temp = Object_intersect(ray, &(objects[i]));

    	if (!isnan(temp) && (distance == -1 || temp < distance)) {
    		distance = temp;
    		object = &(objects[i]);
    	}
    }

    if (object != NULL) {
    	// Raytracer_evaluateShadingModel();
    }
}

__device__ void Raytracer_evaluateShadingModel (color_t * shading_model,
	world_t  * d_w, material_t * material, line_t * ray, float distance)
{
	float ambient = d_w->global_ambient * material->i_ambient,
	 	  intersection[3];

    shading_model->r = material->color[0] * ambient;
    shading_model->b = material->color[1] * ambient;
    shading_model->g = material->color[2] * ambient;

    //finds the intersection point
    Raytracer_findIntersectionPoint(intersection, ray, distance);

    // for (int i = 0; i < d_w->n_lights; ++i) {



    // }
}

__device__ void Raytracer_findIntersectionPoint(float * intersection,
	line_t * ray, float distance)
{
	VECTOR_COPY(intersection, ray->direction);
    VECTOR_SCALE(intersection, distance);
    VECTOR_ADD(intersection, intersection, ray->position);
}

__device__ float Object_intersect (line_t * ray, object_t * object)
{
    switch(object->type) {
        case SPHERE:
            return Sphere_intersect(ray, object->sphere);
        case TRIANGLE:
            return Triangle_intersect(ray, object->triangle);
    }
    return NAN;
}

__device__ float Sphere_intersect (line_t * ray, sphere_t * sphere)
{
    float * center = sphere->center,
    	  radius = sphere->radius,
    	  temp[3], b, c, d, sqrtd;

    //temp = ray origin - center
    VECTOR_SUB(temp, ray->position, center);

    //b = 2 * ray direction . (ray origin - sphere center)
    b = 2 * VECTOR_DOT(ray->direction, temp);

    //c = (ray origin - sphere center) . (ray origin - sphere center)
    // - radius^2
    c = VECTOR_DOT(temp, temp) - pow(radius, 2);

    //d = b^2 - 4 * a * c, a = 1
    d = pow(b, 2) - 4 * c;

    if (d < 0) {
        return NAN;
    } else if (d == 0) {
        return -b / 2;
    } else {
        sqrtd = sqrt(d);
        return fmin(
            (-b + sqrtd) / 2,
            (-b - sqrtd) / 2
        );
    }
}

__device__ float Triangle_intersect (line_t * ray, triangle_t * triangle)
{
    float * direction = ray->direction,
    	  * normal = triangle->normal,
    	  * p0 = triangle->points[0],
    	  intersection[3], u[3], v[3], w[3], temp[3],
          d, numerator, denominator, uu, uv, uw, vv, vw, s, t;

    //denominator = n . direction
    denominator = VECTOR_DOT(normal, direction);

    if (fabs(denominator) < EPSILON) {
        return NAN;
    }

    //numerator = n . (p0 - origin)
    VECTOR_SUB(temp, p0, ray->position);
    numerator = VECTOR_DOT(normal, temp);

    d = numerator / denominator;

    if (d < 0) {
        return NAN;
    }

    Raytracer_findIntersectionPoint(intersection, ray, d);

    VECTOR_SUB(u, triangle->points[1], p0);
    VECTOR_SUB(v, triangle->points[2], p0);
    VECTOR_SUB(w, intersection, p0);

    uu = VECTOR_DOT(u, u);
    uv = VECTOR_DOT(u, v);
    uw = VECTOR_DOT(u, w);
    vv = VECTOR_DOT(v, v);
    vw = VECTOR_DOT(v, w);

    denominator = uv * uv - uu * vv;

    s = (uv * vw - vv * uw) / denominator;
    if (s < 0.0 || s > 1.0) {
        return NAN;
    }

    t = (uv * uw - uu * vw) / denominator;
    if (t < 0.0 || s + t > 1.0) {
        return NAN;
    }

    return d;
}
