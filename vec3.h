#ifndef VEC3H
#define VEC3H
#include "defs.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

class vec3 {

public:
  cpu gpu vec3() {}
  cpu gpu vec3(float e0, float e1, float e2) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
  }
  cpu gpu inline float x() const { return e[0]; }
  cpu gpu inline float y() const { return e[1]; }
  cpu gpu inline float z() const { return e[2]; }
  cpu gpu inline float r() const { return e[0]; }
  cpu gpu inline float g() const { return e[1]; }
  cpu gpu inline float b() const { return e[2]; }

  cpu gpu inline const vec3 &operator+() const { return *this; }
  cpu gpu inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  cpu gpu inline float operator[](int i) const { return e[i]; }
  cpu gpu inline float &operator[](int i) { return e[i]; };

  cpu gpu inline vec3 &operator+=(const vec3 &v2);
  cpu gpu inline vec3 &operator-=(const vec3 &v2);
  cpu gpu inline vec3 &operator*=(const vec3 &v2);
  cpu gpu inline vec3 &operator/=(const vec3 &v2);
  cpu gpu inline vec3 &operator*=(const float t);
  cpu gpu inline vec3 &operator/=(const float t);

  cpu gpu inline float length() const {
    return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  }
  cpu gpu inline float squared_length() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  cpu gpu inline void make_unit_vector();

  float e[3];
};

inline std::istream &operator>>(std::istream &is, vec3 &t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec3 &t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

cpu gpu inline void vec3::make_unit_vector() {
  float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
}

cpu gpu inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

cpu gpu inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

cpu gpu inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

cpu gpu inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

cpu gpu inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

cpu gpu inline vec3 operator/(vec3 v, float t) {
  return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

cpu gpu inline vec3 operator*(const vec3 &v, float t) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

cpu gpu inline float dot(const vec3 &v1, const vec3 &v2) {
  return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

cpu gpu inline vec3 cross(const vec3 &v1, const vec3 &v2) {
  return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
              (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
              (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

cpu gpu inline vec3 &vec3::operator+=(const vec3 &v) {
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

cpu gpu inline vec3 &vec3::operator*=(const vec3 &v) {
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

cpu gpu inline vec3 &vec3::operator/=(const vec3 &v) {
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

cpu gpu inline vec3 &vec3::operator-=(const vec3 &v) {
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

cpu gpu inline vec3 &vec3::operator*=(const float t) {
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

cpu gpu inline vec3 &vec3::operator/=(const float t) {
  float k = 1.0 / t;

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

cpu gpu inline vec3 unit_vector(vec3 v) { return v / v.length(); }

#endif