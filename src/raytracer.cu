
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


	__syncthreads();
}

__device__ void Raytracer_calculatePixelColor (color_t * color, world_t  * d_w,
	line_t * ray)
{
    color_t bg = d_w->bg,
    		shading_model;

    //sets the pixel color to the default background color
    color->r = bg.r;
    color->b = bg.b;
    color->g = bg.g;

    float distance = NAN, temp;
    object_t * object = NULL;

    object_t * objects = d_w->objects;
    for (int i = 0; i < d_w->n_objects; ++i) {
    	temp = Object_intersect(ray, &(objects[i]));

    	if (!isnan(temp) && (isnan(distance) || temp < distance)) {
    		distance = temp;
    		object = &(objects[i]);
    	}
    }

    if (object != NULL) {
    	printf("%f\n", distance);
    	Raytracer_evaluateShadingModel(&shading_model, d_w, object, ray,
    		distance);

        color->r = min(shading_model.r, 255);
        color->b = min(shading_model.b, 255);
        color->g = min(shading_model.g, 255);
    }
}

__device__ void Raytracer_evaluateShadingModel (color_t * shading_model,
	world_t  * d_w, object_t * i_object, line_t * ray, float distance)
{
	material_t material = d_w->materials[i_object->mat];
	float ambient = d_w->global_ambient * material.i_ambient,
	 	  intersection[3], normal[3],
	 	  diffuse, specular, shading;

    shading_model->r = material.color[0] * ambient;
    shading_model->b = material.color[1] * ambient;
    shading_model->g = material.color[2] * ambient;

    //finds the intersection point
    findIntersectionPoint(intersection, ray, distance);

    Vector_normalize(intersection);

    line_t light_ray, reflection_ray;
    light_t * lights = d_w->lights,
    		light;
    object_t * objects = d_w->objects,
    	     * object;
    for (int i = 0; i < d_w->n_lights; ++i) {

    	light = lights[i];

    	VECTOR_COPY(light_ray.position, intersection);
    	VECTOR_SUB(light_ray.direction, light.pos, intersection);
		VECTOR_SCALE(light_ray.direction, 1.0 / VECTOR_LENGTH(light_ray.direction));

    	for (int j = 0; j < d_w->n_objects; ++j) {

    		object = &(objects[i]);


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

        // //computes reflection
        // calculate_pixel_color(&reflection_color, raytracer,
        //     &intersection, &reflection_ray, depth - 1);

        shading = light.i * (material.i_diffuse * diffuse + 
            material.i_specular * specular);

        // color_t color;
        // COLOR_COPY(color, light.color);
        // COLOR_SCALE(color, shading);
        // COLOR_ADD(shading_model, light.color);
        shading_model->r += light.color[0] * shading;
        shading_model->g += light.color[1] * shading;
        shading_model->b += light.color[2] * shading;

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
    float v[3], r[3], r1[3], r2[3];

    VECTOR_COPY(v, ray);
    VECTOR_SCALE(v, -1);

    // R = −L + 2(N.L)N
    VECTOR_COPY(r1, l);
    VECTOR_SCALE(r1, -1);
    VECTOR_COPY(r2, n);
    float temp = VECTOR_DOT(n, l);
    VECTOR_SCALE(r1, 2 * temp);
    VECTOR_ADD(r, r1, r2);

    temp = VECTOR_DOT(v, r);
    return pow(fmax(0, temp), fallout);
}