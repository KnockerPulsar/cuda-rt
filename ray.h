#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        gpu ray() {}
        gpu ray(const vec3& a, const vec3& b) { A = a; B = b; }
        gpu vec3 origin() const       { return A; }
        gpu vec3 direction() const    { return B; }
        gpu vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif