
// Cuda raytracer library functions

#ifndef _h_raytracer
#define _h_raytracer

/******************************
* Creates the rays from a camera on the device
* @param camera Camera on device to create rays for
* @param rays   Pointer to memory allocated on device for rays
******************************/
void Camera_createRays (camera_t* camera, line_t* rays, unsigned int blocks, unsigned int threads);

/******************************
* Kernel for creating the rays from a camera on the device
* @param camera Camera on device to create rays for
* @param rays   Pointer to memory allocated on device for rays
* @param to_do  Number of rays to be created per thread
* @param extra  How many sections need one additional ray
******************************/
__global__ void Camera_createRays_k (camera_t* camera, line_t* rays, unsigned int to_do, unsigned int extra);

#endif
