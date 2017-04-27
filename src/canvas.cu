
#include "canvas.h"

canvas_t* global_canvas;
// Reshape the window event handler
void can_reshape(int w, int h);
// Preform the actual display
void can_display();
// Callback for timing loops
void timer_callback(int t);

// Creates a new canvas
canvas_t* Canvas_create (int height, int width, char* window_name)
{
	// Allocate resources for canvas
	canvas_t* result = (canvas_t*) malloc(sizeof(canvas_t));
	// Allocate memory for pixels
	result->pixels = (color_t*) malloc(sizeof(color_t) * height * width);

	// Store given data for canvas
	result->width = width;
	result->height = height;
	// Copy title of window
	strcpy(result->window_name, window_name);
	// Set default frame delay
	result->delay = DEFAULT_REDRAW_DELAY;
	result->message[0] = '\0';

	return result;
}

// Sets the pixel at the current location to the specified value
void Canvas_setPixel(int x, int y, int r, int g, int b)
{
   color_t* t;

   t = &(global_canvas->pixels[y * global_canvas->width + x]);
   t->r = r;
   t->g = g;
   t->b = b;
}

// Set the loop function for the canvas.
void Canvas_setRenderFunction (canvas_t* can, void (func)(), unsigned int time)
{
	// Save the redraw animate functionm
	can->animate = func;
	// Store the redraw delay
	can->delay = time;
}

// Runs the render loop of the specified canvas
void Canvas_startLoop (canvas_t* can, int argc, char** argv)
{
	global_canvas = can;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(global_canvas->height, global_canvas->width);
	glutInitWindowPosition(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y);
	glutCreateWindow(global_canvas->window_name);
	glutDisplayFunc(can_display);
	glutReshapeFunc(can_reshape);
	timer_callback(global_canvas->delay);
	glutMainLoop();
}

// Frees resources allocated for a canvas
void Canvas_free (canvas_t* target)
{
	// Free memory allocated for pixels
	free(target->pixels);
	// Free memory allocated for canvas object
	free(target);
}

// Callback for timing loops
void timer_callback(int t)
{
    global_canvas->animate();
    glutPostRedisplay();
    glutTimerFunc(global_canvas->delay, timer_callback, 0);
}

// Redraws the canvas
void can_display()
{
   int x_itr, y_itr;
   color_t* curr;

   // Redraw the pixels
   glBegin(GL_POINTS);
   curr = global_canvas->pixels;
   for (y_itr = 0; y_itr < global_canvas->height; y_itr++) {
      for (x_itr = 0; x_itr < global_canvas->width; x_itr++, curr++) {
         glColor3ub(curr->r, curr->g, curr->b);
         glVertex2f(x_itr, y_itr);
      }
   }
   glEnd();
   // Display the message

	glColor3ub(255, 255, 255);
	// Set the location of the message
	glRasterPos2i(5, global_canvas->height - 15);
	// Draw the chracters
	for (int character_iterator = 0; global_canvas->message[character_iterator] != '\0'; character_iterator++) {
		glutBitmapCharacter(MESSAGE_FONT, global_canvas->message[character_iterator]);
	}

   //display_messages();
   glutSwapBuffers();
}

// Reshape the window event handler
void can_reshape(int w, int h)
{
   // Set viewport size
   glViewport(0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   // Set relative ortho
   glOrtho(0, global_canvas->width, 0, global_canvas->height, -1, 1);  
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}
