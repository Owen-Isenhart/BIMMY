#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class OrbitCamera {
 public:
  float yawDeg = -35.0f;
  float pitchDeg = -25.0f;
  float distance = 10.0f;
  glm::vec3 target = glm::vec3(0.0f, 1.0f, 0.0f);
  bool firstPerson = false;
  glm::vec3 firstPersonPosition = glm::vec3(0.0f, 1.8f, 6.0f);

  glm::vec3 ForwardVector() const {
    const float yaw = glm::radians(yawDeg);
    const float pitch = glm::radians(pitchDeg);
    return glm::normalize(glm::vec3(
      cosf(pitch) * cosf(yaw),
      sinf(pitch),
      cosf(pitch) * sinf(yaw)));
  }

  glm::vec3 Position() const {
    if (firstPerson) {
      return firstPersonPosition;
    }

    return target - ForwardVector() * distance;
  }

  glm::mat4 ViewMatrix() const {
    if (firstPerson) {
      const glm::vec3 eye = firstPersonPosition;
      return glm::lookAt(eye, eye + ForwardVector(), glm::vec3(0, 1, 0));
    }

    return glm::lookAt(Position(), target, glm::vec3(0, 1, 0));
  }

  void EnableFirstPerson() {
    if (firstPerson) {
      return;
    }

    firstPersonPosition = Position();
    firstPerson = true;
  }

  void DisableFirstPerson() {
    if (!firstPerson) {
      return;
    }

    firstPerson = false;
    target = firstPersonPosition + ForwardVector() * distance;
  }

  void Orbit(float dx, float dy) {
    yawDeg += dx;
    pitchDeg += dy;
    if (pitchDeg > 85.0f) pitchDeg = 85.0f;
    if (pitchDeg < -85.0f) pitchDeg = -85.0f;
  }

  void Zoom(float delta) {
    if (firstPerson) {
      return;
    }

    distance *= (1.0f - delta * 0.1f);
    if (distance < 1.0f) distance = 1.0f;
    if (distance > 120.0f) distance = 120.0f;
  }
};
