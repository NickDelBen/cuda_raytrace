
#include "raytracer.h"

__host__ void CudaCheckError()
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }
}

void Raytracer(COLOR * d_f, line_t * d_r, world_t * w, int size, int blocks,
    int threads, int max_reflections)
{
    int b_work   = size / blocks,
        t_work   = b_work / threads,
        r_size   = sizeof(line_t) * size,
        b_r_size = sizeof(line_t) * b_work,
        b_f_size = sizeof(COLOR) * CHANNELS * b_work,
        b_b_size = sizeof(float) * b_work,
        b_w_size = 0;

    // Copy animated world to device.
    world_t * d_w = World_toDevice(w, &b_w_size);

    line_t * rays;
    cudaMalloc(&rays, r_size);
    cudaMemcpy(rays, d_r, r_size, cudaMemcpyDeviceToDevice);

    printf("Allocating %d bytes per block\n",
        b_r_size + b_f_size + b_w_size + b_b_size);
    Raytracer_trace<<<blocks, threads, b_r_size + b_f_size + b_w_size + b_b_size>>>
        (rays, d_f, d_w, b_w_size, b_work, t_work, max_reflections);
    CudaCheckError();
    cudaDeviceSynchronize();
        
    World_freeDevice(d_w);

    cudaFree(rays);
}

__global__ void Raytracer_trace (line_t * d_r, COLOR * d_f, world_t * w,
    int w_size, int b_work, int t_work, int max_reflections)
{
    int t_offset = threadIdx.x * t_work,
        offset   = blockIdx.x * b_work + t_offset;

    // ** Add world to shared memory for faster access time ** //
    extern __shared__ uint8_t smem[];

    // Assign shared memory locations to the world, rays array and frame array.
    world_t * d_w = World_toShared((void *) smem, w);
    line_t  * rays = (line_t *)(smem + w_size);
    COLOR   * frame = (COLOR *)&rays[b_work];
    float   * reflectivities = (float *)&frame[b_work * CHANNELS];

    // Copy from global memory to shared memory.
    memcpy(&rays[t_offset], &d_r[offset], sizeof(line_t) * t_work);
    memcpy(&frame[t_offset * CHANNELS], &d_f[offset * CHANNELS],
        sizeof(COLOR) * CHANNELS * t_work);

    // Process all the pixels assigned to this thread
    for (int i = t_offset; i < t_offset + t_work; ++i) {
        reflectivities[i] = Raytracer_calculatePixelColor(&frame[i * CHANNELS],
            d_w, &rays[i]);
    }

    float reflectivity;
    COLOR reflection_color[CHANNELS];
    for (int i = 1; i < max_reflections; ++i) {
        // Process all the pixels assigned to this thread
        for (int j = t_offset; j < t_offset + t_work; ++j) {

            if (!isnan(reflectivities[j])) {
                reflectivity = Raytracer_calculatePixelColor(reflection_color,
                    d_w, &rays[j]);

                COLOR_SCALE(reflection_color, reflectivities[j]);
                COLOR_ADD(&frame[i * CHANNELS], &frame[i * CHANNELS], reflection_color);

                reflectivities[j] = reflectivity;
            }
        }
    }

    // Copy the results of the trace on the frame tile to the global memory.
    memcpy(&d_r[offset], &rays[t_offset], sizeof(line_t) * t_work);
    memcpy(&d_f[offset * CHANNELS], &frame[t_offset * CHANNELS],
        sizeof(COLOR) * CHANNELS * t_work);
}

__device__ float Raytracer_calculatePixelColor (COLOR * color, world_t * d_w,
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
        Raytracer_evaluateShadingModel(color, d_w, object, ray, distance);
        return d_w->materials[object->mat].reflectivity;
    }

    return NAN;
}

__device__ void Raytracer_evaluateShadingModel (COLOR * color,
    world_t  * d_w, object_t * i_object, line_t * ray, float distance)
{
    COLOR diffuse[CHANNELS], specular[CHANNELS], shading[CHANNELS];
    material_t material = d_w->materials[i_object->mat];
    float ambient = d_w->global_ambient * material.i_ambient,
          intersection[DSPACE], normal[DSPACE],
          distance = NAN, temp;

    VECTOR_SCALE(color, material.color, ambient);

    //finds the intersection point
    findIntersectionPoint(intersection, ray, distance);

    //retrieves the normal at the point of intersection
    Object_normal(normal, i_object, intersection);

    line_t shadow_ray;
    light_t light;
    object_t * object;
    for (int i = 0; i < d_w->n_lights; ++i) {

        light = (d_w->lights)[i];

        VECTOR_COPY(shadow_ray.position, intersection);
        VECTOR_SUB(shadow_ray.direction, light.pos, intersection);
        Vector_normalize(shadow_ray.direction);

        for (int j = 0; j < d_w->n_objects; ++j) {

            object = &((d_w->objects)[i]);

            if (object == i_object) {
                continue;
            }

            temp = Object_intersect(&shadow_ray, object);

            if (!isnan(temp) && (isnan(distance) || temp < distance)) {
                distance = temp;
            }
        }

        if (!isnan(distance)) {
            COLOR_SCALE(diffuse, material.color, light.i * material.i_diffuse * 
                Raytracer_diffuse(normal, shadow_ray.direction));
            COLOR_SCALE(specular, material.color, light.i * material.i_specular *
                Raytracer_specular(ray->direction,normal,
                    shadow_ray.direction, material.specular_power))
                    
            COLOR_ADD(color, color, diffuse);
            COLOR_ADD(color, color, specular);
        }
    }

    VECTOR_COPY(ray->position, intersection);
    findReflectedRay(ray->direction, ray->direction, normal);
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

    temp = VECTOR_DOT(n, l);
    VECTOR_SCALE(r2, n, 2 * temp);
    VECTOR_ADD(r, r1, r2);

    temp = VECTOR_DOT(v, r);
    return pow(fmax(0, temp), fallout);
}