#pragma once

#include <limits>
#include <optional>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "bimmy_types.hpp"

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

inline Ray BuildRayFromScreen(
  double mouseX,
  double mouseY,
  int width,
  int height,
  const glm::mat4& view,
  const glm::mat4& proj) {
  const float x = (2.0f * static_cast<float>(mouseX)) / static_cast<float>(width) - 1.0f;
  const float y = 1.0f - (2.0f * static_cast<float>(mouseY)) / static_cast<float>(height);

  const glm::vec4 rayClipNear = glm::vec4(x, y, -1.0f, 1.0f);
  const glm::vec4 rayClipFar = glm::vec4(x, y, 1.0f, 1.0f);

  const glm::mat4 invVP = glm::inverse(proj * view);
  glm::vec4 rayWorldNear = invVP * rayClipNear;
  glm::vec4 rayWorldFar = invVP * rayClipFar;

  rayWorldNear /= rayWorldNear.w;
  rayWorldFar /= rayWorldFar.w;

  const glm::vec3 origin = glm::vec3(rayWorldNear);
  const glm::vec3 direction = glm::normalize(glm::vec3(rayWorldFar - rayWorldNear));

  return {origin, direction};
}

inline bool IntersectRayAABB(const Ray& ray, const glm::vec3& aabbMin, const glm::vec3& aabbMax, float& tOut) {
  float tMin = 0.0f;
  float tMax = std::numeric_limits<float>::max();

  for (int i = 0; i < 3; ++i) {
    if (fabsf(ray.direction[i]) < 1e-6f) {
      if (ray.origin[i] < aabbMin[i] || ray.origin[i] > aabbMax[i]) {
        return false;
      }
      continue;
    }

    float invD = 1.0f / ray.direction[i];
    float t1 = (aabbMin[i] - ray.origin[i]) * invD;
    float t2 = (aabbMax[i] - ray.origin[i]) * invD;

    if (t1 > t2) std::swap(t1, t2);

    tMin = t1 > tMin ? t1 : tMin;
    tMax = t2 < tMax ? t2 : tMax;

    if (tMin > tMax) {
      return false;
    }
  }

  tOut = tMin;
  return true;
}

inline std::optional<std::uint32_t> PickComponent(const Ray& ray, const std::vector<Component>& components) {
  float nearest = std::numeric_limits<float>::max();
  std::optional<std::uint32_t> picked = std::nullopt;

  for (const Component& c : components) {
    glm::vec3 aabbMin, aabbMax;
    c.WorldAABB(aabbMin, aabbMax);
    float t = 0.0f;
    if (IntersectRayAABB(ray, aabbMin, aabbMax, t)) {
      if (t < nearest) {
        nearest = t;
        picked = c.id;
      }
    }
  }

  return picked;
}
