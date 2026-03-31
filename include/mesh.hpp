#pragma once

#include <cstdint>
#include <vector>

#include <glad/gl.h>
#include <glm/glm.hpp>

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 uv;
};

struct MeshCpu {
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
};

class GpuMesh {
 public:
  GpuMesh() = default;
  GpuMesh(const MeshCpu& mesh);
  ~GpuMesh();

  GpuMesh(const GpuMesh&) = delete;
  GpuMesh& operator=(const GpuMesh&) = delete;

  GpuMesh(GpuMesh&& other) noexcept;
  GpuMesh& operator=(GpuMesh&& other) noexcept;

  void Draw() const;

 private:
  GLuint vao_ = 0;
  GLuint vbo_ = 0;
  GLuint ebo_ = 0;
  GLsizei indexCount_ = 0;
};

MeshCpu CreateCuboidMesh();
MeshCpu CreateCylinderMesh(int segments);
MeshCpu CreatePrismMesh();
MeshCpu CreatePlaneMesh();
