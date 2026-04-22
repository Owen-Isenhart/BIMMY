#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/gl.h>

constexpr float kPi = 3.14159265359f;

enum class GeometryType {
  Cuboid,
  Cylinder,
  Prism,
};

enum class MaterialType {
  SheetMetal,
  Grass,
  Concrete,
  RustedMetal,
  Brick,
  Roof,
  Wood,
};

struct MaterialDefinition {
  std::string name;
  float costPerUnitVolume;
  GLuint albedoMap = 0;
  GLuint normalMap = 0;
  GLuint metallicMap = 0;
  GLuint roughnessMap = 0;
  GLuint aoMap = 0;
  GLuint heightMap = 0;
  float roughnessMultiplier = 1.0f;
  float metallicMultiplier = 1.0f;
};

std::unordered_map<MaterialType, MaterialDefinition>& MutableMaterialCatalog();
const std::unordered_map<MaterialType, MaterialDefinition>& MaterialCatalog();

struct Transform {
  glm::vec3 position = glm::vec3(0.0f);
  glm::vec3 rotationEulerDeg = glm::vec3(0.0f);
  glm::vec3 scale = glm::vec3(1.0f);

  glm::mat4 ModelMatrix() const {
    glm::mat4 m = glm::translate(glm::mat4(1.0f), position);
    m = glm::rotate(m, glm::radians(rotationEulerDeg.x), glm::vec3(1, 0, 0));
    m = glm::rotate(m, glm::radians(rotationEulerDeg.y), glm::vec3(0, 1, 0));
    m = glm::rotate(m, glm::radians(rotationEulerDeg.z), glm::vec3(0, 0, 1));
    return glm::scale(m, scale);
  }
};

struct Component {
  std::uint32_t id = 0;
  GeometryType geometry = GeometryType::Cuboid;
  MaterialType material = MaterialType::Concrete;
  glm::vec3 dimensions = glm::vec3(1.0f); // Cuboid: xyz, Cylinder: r/h, Prism: w/h/d
  Transform transform;

  float Volume() const {
    const glm::vec3 absScale = glm::abs(transform.scale);
    switch (geometry) {
      case GeometryType::Cuboid: {
        glm::vec3 d = dimensions * absScale;
        return d.x * d.y * d.z;
      }
      case GeometryType::Cylinder: {
        const float radius = dimensions.x * (absScale.x + absScale.z) * 0.5f;
        const float height = dimensions.y * absScale.y;
        return kPi * radius * radius * height;
      }
      case GeometryType::Prism: {
        const float width = dimensions.x * absScale.x;
        const float height = dimensions.y * absScale.y;
        const float depth = dimensions.z * absScale.z;
        return 0.5f * width * height * depth;
      }
      default:
        return 0.0f;
    }
  }

  float Cost() const {
    const auto& catalog = MaterialCatalog();
    const auto it = catalog.find(material);
    if (it == catalog.end()) {
      return 0.0f;
    }
    return Volume() * it->second.costPerUnitVolume;
  }

  glm::vec3 LocalHalfExtents() const {
    switch (geometry) {
      case GeometryType::Cuboid:
        return dimensions * 0.5f;
      case GeometryType::Cylinder:
        return glm::vec3(dimensions.x, dimensions.y * 0.5f, dimensions.x);
      case GeometryType::Prism:
        return glm::vec3(dimensions.x * 0.5f, dimensions.y * 0.5f, dimensions.z * 0.5f);
      default:
        return glm::vec3(0.5f);
    }
  }

  // Axis-aligned bounds used for fast picking.
  void WorldAABB(glm::vec3& outMin, glm::vec3& outMax) const {
    const glm::vec3 he = LocalHalfExtents();
    const glm::mat4 m = transform.ModelMatrix();

    glm::vec3 corners[8] = {
      {-he.x, -he.y, -he.z},
      { he.x, -he.y, -he.z},
      {-he.x,  he.y, -he.z},
      { he.x,  he.y, -he.z},
      {-he.x, -he.y,  he.z},
      { he.x, -he.y,  he.z},
      {-he.x,  he.y,  he.z},
      { he.x,  he.y,  he.z},
    };

    outMin = glm::vec3(1e9f);
    outMax = glm::vec3(-1e9f);

    for (const glm::vec3& c : corners) {
      const glm::vec3 w = glm::vec3(m * glm::vec4(c, 1.0f));
      outMin = glm::min(outMin, w);
      outMax = glm::max(outMax, w);
    }
  }
};

struct BomTotals {
  std::unordered_map<MaterialType, float> volumeByMaterial;
  std::unordered_map<MaterialType, float> costByMaterial;
  float grandTotal = 0.0f;
};

inline BomTotals ComputeBom(const std::vector<Component>& components) {
  BomTotals totals;
  for (const Component& c : components) {
    const float volume = c.Volume();
    const float cost = c.Cost();
    totals.volumeByMaterial[c.material] += volume;
    totals.costByMaterial[c.material] += cost;
    totals.grandTotal += cost;
  }
  return totals;
}
