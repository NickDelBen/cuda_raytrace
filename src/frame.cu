
#include "frame.h"

// Initialize the frame
__global__ void Frame_init (color_t *frame, int size)
{
	int b_size 	= size / gridDim.x,
		t_size 	= b_size / blockDim.x,
		offset 	= blockIdx.x * b_size + threadIdx.x * t_size;

	color_t *thread_pixles = (color_t*)malloc(sizeof(color_t) * t_size);

	for (int i = 0; i < t_size; ++i) {
		thread_pixles[i].r = 0;
		thread_pixles[i].g = 0;
		thread_pixles[i].b = 0;
	}

	memcpy(&frame[offset], &thread_pixles, sizeof(color_t) * t_size);
}
