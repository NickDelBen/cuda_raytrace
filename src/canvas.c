
#include "canvas.h"

canvas_t* global_canvas;
// Reshape the window event handler
void reshape(int w, int h);
// Preform the actual display
void can_display();
// Callback for timing loops
void timer_callback(unsigned int t);

// Creates a new canvas
void Canvas_create (int height, int width, char* window_name)
{
	// Allocate resources for canvas
	canvas_t* result = malloc(sizeof(canvas_t));
	// Allocate memory for pixels
	result->pixels = malloc(sizeof(color_t) * height * width);

	// Store given data for canvas
	result->width = width;
	result->height = height;
	// Copy title of window
	strcpy(result->window_name, window_name);
	// Set default frame delay
	result->delay = DEFAULT_REDRAW_DELAY;
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
	glutInitWindowSize(can->height, can->width);
	glutInitWindowPosition(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y);
	glutCreateWindow(can->window_name);
	glutDisplayFunc(can_display);
	glutReshapeFunc(can_reshape);
	timer_callback(can->delay);
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
