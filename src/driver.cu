
#include "driver.h"

int main(int argc, char **argv)
{

	if (argc != 2) {
		printf("Please provide the scene file descriptor as an argument.\n");
		return EXIT_FAILURE;
	}

	FILE *fp = fopen(argv[1], "r");

	if (!fp) {
		printf("Unable to open file.\n");
		return EXIT_FAILURE;
	}

	camera_t *c = Camera_read(fp);
	printf("Read and Created camera on host\n");

	world_t *w = World_read(fp);
	printf("Read and created world on host\n");
	printf("Read world background=(%hu, %hu, %hu)   gloabl_ambient=%f\n", w->bg[0], w->bg[1], w->bg[2], w->global_ambient);
	
	fclose(fp);

	camera_t *d_c = Camera_toDevice(c);
	printf("Copied camera to device\n");

	int size = c->height * c->width;

	line_t* d_r;
	cudaMalloc(&d_r, size);
	Camera_createRays(d_c, d_r, BLOCKS, THREADS);
	printf("Created rays from camera on device\n");
	
	// Create frame
	color_t *f, *d_f;

	f = (color_t*)malloc(sizeof(color_t) * size);
	cudaMalloc(&d_f, sizeof(color_t) * size);

	// while (true) {

		trace(f, d_f, d_r, w, size);
		// paint(f);
		// animate(w);

	// }

	cudaFree(d_r);
	printf("Freed frame on host\n");

	free(f);
	printf("Freed frame on device\n");

	cudaFree(d_r);
	printf("Freed rays on device\n");

	Camera_freeDevice(d_c);
	printf("Freed camera on host\n");

	Camera_freeHost(c);
	printf("Freed camera on host\n");

	World_freeHost(w);
	printf("Freed world on host\n");

	return EXIT_SUCCESS;
}

void trace(color_t *f, color_t * d_f, line_t *d_r, world_t *w, int size) {

	world_t *d_w = World_toDevice(w);
	printf("Copied world to device\n");

	// Create and initialize frame
	Frame_init<<<BLOCKS, THREADS>>>(d_f, size);

	// for (int i = 0; i < MAX_REFLECTIONS; ++i) {

	// }

	World_freeDevice(d_w);
	printf("Freed world on device\n");

	// Copy frame to host
	cudaMemcpy(f, d_f, sizeof(color_t) * size, cudaMemcpyDeviceToHost);

}