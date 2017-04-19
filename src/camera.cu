
#include "camera.h"

// Reads the details of a camera from the specified file
camera_t* Camera_read (FILE* file)
{
	camera_t* result = (camera_t*) malloc(sizeof(camera_t));

	fscanf(file, "CAM %f %f %f %f %f %f %f %f %f\n", &(result->position[0]), &(result->position[1]), &(result->position[2]),
		&(result->top_right[0]), &(result->top_right[1]), &(result->top_right[2]), 
		&(result->bottom_left[0]), &(result->bottom_left[1]), &(result->bottom_left[2]));

	result->width = ;
	result->height = ;

	return result;
}

// Copies the specified camera to the device
camera_t* Camera_toDevice (camera_t* source)
{
	camera_t* result;

	// Allocate space for camera on device
	cudaMalloc(&result, sizeof(camera_t));
	// Copy camera to device
	cudaMemcpy(result, source, sizeof(camera_t), cudaMemcpyHostToDevice);

	return result;
}

// Frees resources allocated for a camera on the host
void Camera_freeHost (camera_t* camera)
{
	free(camera);
}

// Frees resources allocated for a camera on the device
void Camera_freeDevice (camera_t* camera)
{
	cudaFree(camera);
}
