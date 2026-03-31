#pragma once

#include <unordered_map>
#include <vector>

#include <glad/gl.h>
#include <glm/glm.hpp>

#include "bimmy_types.hpp"
#include "camera.hpp"
#include "mesh.hpp"

class Renderer {
 public:
  bool Initialize();
  void Shutdown();

  void DrawScene(const std::vector<Component>& components, std::uint32_t selectedId, const OrbitCamera& camera, int viewportWidth, int viewportHeight, bool drawFloor);

 private:
  bool BuildProgram();

  GLuint program_ = 0;
  GpuMesh cuboid_;
  GpuMesh cylinder_;
  GpuMesh prism_;
  GpuMesh plane_;
};
