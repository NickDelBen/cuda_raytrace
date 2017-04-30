
#include "raytracer.h"

__host__ void CudaCheckError()
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }
}

void Raytracer(COLOR * d_f, line_t * d_r, world_t * w, int size,
	int blocks, int threads, int max_reflections)
{
	int b_work = size / blocks,
		t_work = b_work / threads,
		r_size = sizeof(line_t) * b_work,
		f_size = sizeof(COLOR) * CHANNELS * b_work;

	// Copy animated world to device.
	world_t * d_w = World_toDevice(w);

	line_t * rays;
	cudaMalloc(&rays, sizeof(line_t) * size);
	cudaMemcpy(rays, d_r, sizeof(line_t) * size, cudaMemcpyDeviceToDevice);

	// Traces rays bounces.
	for (int i = 0; i < max_reflections; ++i) {
		Raytracer_trace<<<blocks, threads, r_size + f_size>>>(rays, d_f, d_w,
			size, b_work, t_work);
        CudaCheckError();
	}

	// Frees world from device memory.
	World_freeDevice(d_w);
}

__global__ void Raytracer_trace (line_t * d_r, COLOR * d_f, world_t * d_w,
	int size, int b_work, int t_work)
{
	int t_offset 	= threadIdx.x * t_work,
		offset 		= blockIdx.x * b_work + t_offset;

	// ** Add world to shared memory for faster access time ** //

	extern __shared__ float smem[];

	// Assign shared memory locations to the rays and frame arrays.
	line_t * rays = (line_t *)smem;
	COLOR * frame = (COLOR *)&rays[b_work],
			result[CHANNELS];

	// Copy from global memory to shared memory.
	memcpy(&rays[t_offset], &d_r[offset], sizeof(line_t) * t_work);
	memcpy(&frame[t_offset], &d_f[offset], sizeof(COLOR) * CHANNELS * t_work);

	// Process all the pixels assigned to this thread
	for (int i = t_offset; i < t_offset + t_work; ++i) {
		Raytracer_calculatePixelColor(result, d_w, &rays[i]);
		// VECTOR_ADD(&frame[i], &frame[i], &result);
	}

	// Copy the results of the trace on the frame tile to the global memory.
	memcpy(&d_r[offset], &rays[t_offset], sizeof(line_t) * t_work);
	memcpy(&d_f[offset], &frame[t_offset], sizeof(COLOR) * CHANNELS * t_work);
}

__device__ void Raytracer_calculatePixelColor (COLOR * color, world_t * d_w,
	line_t * ray)
{
    //sets the pixel color to the default background color
    COLOR_COPY(color, d_w->bg);

    float distance = NAN, temp;
    object_t * object = NULL;

    for (int i = 0; i < d_w->n_objects; ++i) {
    	temp = Object_intersect(ray, &((d_w->objects)[i]));

    	if (!isnan(temp) && (isnan(distance) || temp < distance)) {
    		distance = temp;
    		object = &((d_w->objects)[i]);
    	}
    }

    if (object != NULL) {
    	Raytracer_evaluateShadingModel(color, d_w, object, ray,
    		distance);
    }
}

__device__ void Raytracer_evaluateShadingModel (COLOR * shading_model,
	world_t  * d_w, object_t * i_object, line_t * ray, float distance)
{
    COLOR temp[CHANNELS];
	material_t material = d_w->materials[i_object->mat];
	float ambient = d_w->global_ambient * material.i_ambient,
	 	  intersection[DSPACE], normal[DSPACE],
	 	  diffuse, specular, shading;

    VECTOR_SCALE(shading_model, material.color, ambient);

    //finds the intersection point
    findIntersectionPoint(intersection, ray, distance);

    line_t light_ray, reflection_ray;
    light_t light;
    object_t * object;
    for (int i = 0; i < d_w->n_lights; ++i) {

    	light = (d_w->lights)[i];

    	VECTOR_COPY(light_ray.position, intersection);
    	VECTOR_SUB(light_ray.direction, light.pos, intersection);
		Vector_normalize(light_ray.direction);

    	for (int j = 0; j < d_w->n_objects; ++j) {

    		object = &((d_w->objects)[i]);

    		if (object == i_object) {
    			continue;
    		}

    		if (Object_intersect(&light_ray, object) > -1) {
    			goto SKIP_SHADING;
    		}
    	}

        //retrieves the normal at the point of intersection
        Object_normal(normal, object, intersection);

        VECTOR_COPY(reflection_ray.position, intersection);
        findReflectedRay(reflection_ray.direction, ray->direction, normal);

        //computes the shading
        diffuse = Raytracer_diffuse(normal, light_ray.direction);
        specular = Raytracer_specular(ray->direction, normal, light_ray.direction,
        	material.specular_power);

        shading = light.i * (material.i_diffuse * diffuse + 
            material.i_specular * specular);

        VECTOR_SCALE(temp, light.color, shading);
        VECTOR_ADD(shading_model, shading_model, temp);

	    SKIP_SHADING:
	    continue;
    }
}

__device__ float Raytracer_diffuse(float * n, float * l)
{
    //max(0, light source vector . normal vector
    float diffuse = VECTOR_DOT(l, n);
    return fmax(0, diffuse);
}

__device__ float Raytracer_specular(float * ray, float * n,
    float * l, float fallout)
{
    float v[DSPACE], r[DSPACE], r1[DSPACE], r2[DSPACE], 
    	  temp;

    VECTOR_SCALE(v, ray, -1);

    // R = −L + 2(N.L)N
    VECTOR_SCALE(r1, l, -1);
    VECTOR_COPY(r2, n);
    temp = VECTOR_DOT(n, l);
    VECTOR_SCALE(r1, 2 * temp);
    VECTOR_ADD(r, r1, r2);

    temp = VECTOR_DOT(v, r);
    return pow(fmax(0, temp), fallout);
}