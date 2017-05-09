
#include "camera.h"

// Reads the details of a camera from the specified file
camera_t * Camera_read (FILE * file)
{
	camera_t * result = (camera_t *) malloc(sizeof(camera_t));

	fscanf(file, "CAM POINTS "
		"(%f, %f, %f), "
		"(%f, %f, %f), "
		"(%f, %f, %f), "
		"(%f, %f, %f)\n",
		&(result->bottom_left[X]), &(result->bottom_left[Y]), &(result->bottom_left[Z]),
		&(result->top_left[X]), &(result->top_left[Y]), &(result->top_left[Z]),
		&(result->top_right[X]), &(result->top_right[Y]), &(result->top_right[Z]), 
		&(result->bottom_right[X]), &(result->bottom_right[Y]), &(result->bottom_right[Z]));

	// Get the direction vectors between the corners
	VECTOR_SUB(result->comp_vert, result->top_left, result->bottom_left);
	VECTOR_SUB(result->comp_horiz, result->bottom_right, result->bottom_left);
	// Store the height and width of imaging plane
	result->width = floor(VECTOR_LENGTH(result->comp_horiz));
	result->height = floor(VECTOR_LENGTH(result->comp_vert));

	Camera_calculateVectors(result);

	return result;
}

// Creates the rays from a camera on the device
__global__ void Camera_createRays (camera_t * d_camera, line_t * rays)
{
	unsigned int thread_count, thread_real, extra, to_do, pixels, x, y;
	float vert[DSPACE], horiz[DSPACE];
	line_t *cur;

	pixels = (((unsigned int) d_camera->width) * ((unsigned int) d_camera->height));
	// Calculate number of threads kernel is using
	thread_count = gridDim.x * blockDim.x;
	// Find index of current thread
	thread_real = blockDim.x * blockIdx.x + threadIdx.x;
	// Calculate the amount of threads that must do an additional pixel
	extra = pixels % thread_count;
	// Calculate the amount of pixels this thread has to take care of
	to_do = pixels / thread_count;

	// Calculate the start and end pixel indices for this thread
	pixels = thread_real * to_do;
	if (thread_real >= extra) {
		pixels += extra;
	} else {
		pixels += thread_real;
		to_do += 1;
	}
	to_do += pixels;

	while (pixels < to_do) {
		// Calculate coordinate of target pixel in imaging plane
		y = pixels / ((unsigned int) d_camera->width);
		x = pixels % ((unsigned int) d_camera->width);
		cur = rays + pixels;
		// Calculate the components of the position
		VECTOR_SCALE_3(vert, d_camera->comp_vert, y);
		VECTOR_SCALE_3(horiz, d_camera->comp_horiz, x);	
		// Add the components to get true position
		VECTOR_ADD(cur->position, vert, horiz);
		++pixels;
	}
}

// Copies the specified camera to the device
camera_t * Camera_toDevice (camera_t * source)
{
	camera_t * result;

	// Allocate space for camera on device
	cudaMalloc(&result, sizeof(camera_t));
	// Copy camera to device
	cudaMemcpy(result, source, sizeof(camera_t), cudaMemcpyHostToDevice);

	return result;
}

// Calculate the camera direction vectors and normal
void Camera_calculateVectors (camera_t * cam)
{
	// Get the direction vectors between the corners
	VECTOR_SUB(cam->comp_vert, cam->top_left, cam->bottom_left);
	VECTOR_SUB(cam->comp_horiz, cam->bottom_right, cam->bottom_left);

	// Turn the direction vectors into unit vectors
	Vector_normalize(cam->comp_vert);
	Vector_normalize(cam->comp_horiz);
	// Find the normal
	VECTOR_CROSSPRODUCT(cam->normal, cam->comp_horiz, cam->comp_vert);
}

// Frees resources allocated for a camera on the host
void Camera_freeHost (camera_t * camera)
{
	free(camera);
}

// Frees resources allocated for a camera on the device
void Camera_freeDevice (camera_t * camera)
{
	cudaFree(camera);
}
