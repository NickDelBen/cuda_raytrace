
#include "triangle.h"

// Reads triangle data from the specified file and sets specified triangle
void Triangle_readTo (FILE* file, triangle_t* triangle)
{
	fscanf(file, "%f %f %f %f %f %f %f %f %f\n",
		&(triangle->points[0][0]), &(triangle->points[0][1]), &(triangle->points[0][2]),
		&(triangle->points[1][0]), &(triangle->points[1][1]), &(triangle->points[1][2]),
		&(triangle->points[2][0]), &(triangle->points[2][1]), &(triangle->points[2][2]));

	float u[3], v[3];
	VECTOR_SUB(u, triangle->points[1], triangle->points[0]);
    VECTOR_SUB(v, triangle->points[2], triangle->points[0]);
    VECTOR_CROSSPRODUCT(triangle->normal, u, v);
    Vector_normalize(triangle->normal);
}

// Find intersection points between ray and triangle.
__device__ float Triangle_intersect (line_t * ray, triangle_t * triangle)
{
    float * direction = ray->direction,
    	  * normal = triangle->normal,
    	  * p0 = triangle->points[0],
    	  intersection[3], u[3], v[3], w[3], temp[3],
          d, numerator, denominator, uu, uv, uw, vv, vw, s, t;

    //denominator = n . direction
    denominator = VECTOR_DOT(normal, direction);

    if (fabs(denominator) < EPSILON) {
        return NAN;
    }

    //numerator = n . (p0 - origin)
    VECTOR_SUB(temp, p0, ray->position);
    numerator = VECTOR_DOT(normal, temp);

    d = numerator / denominator;

    if (d < 0) {
        return NAN;
    }

    findIntersectionPoint(intersection, ray, d);

    VECTOR_SUB(u, triangle->points[1], p0);
    VECTOR_SUB(v, triangle->points[2], p0);
    VECTOR_SUB(w, intersection, p0);

    uu = VECTOR_DOT(u, u);
    uv = VECTOR_DOT(u, v);
    uw = VECTOR_DOT(u, w);
    vv = VECTOR_DOT(v, v);
    vw = VECTOR_DOT(v, w);

    denominator = uv * uv - uu * vv;

    s = (uv * vw - vv * uw) / denominator;
    if (s < 0.0 || s > 1.0) {
        return NAN;
    }

    t = (uv * uw - uu * vw) / denominator;
    if (t < 0.0 || s + t > 1.0) {
        return NAN;
    }

    return d;
}