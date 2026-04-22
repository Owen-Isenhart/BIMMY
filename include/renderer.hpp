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
  struct Settings {
    glm::vec3 pointLightPosition = glm::vec3(7.0f, 10.0f, 8.0f);
    glm::vec3 pointLightColor = glm::vec3(1.0f, 0.96f, 0.9f);
    float pointLightIntensity = 90.0f;

    glm::vec3 directionalDirection = glm::normalize(glm::vec3(-0.6f, -1.0f, -0.4f));
    glm::vec3 directionalColor = glm::vec3(1.0f, 0.98f, 0.95f);
    float directionalIntensity = 2.8f;

    glm::vec3 skyAmbientColor = glm::vec3(0.28f, 0.36f, 0.50f);
    glm::vec3 groundAmbientColor = glm::vec3(0.20f, 0.16f, 0.12f);
    float ambientIntensity = 0.50f;

    float exposure = 1.25f;
    float bloomStrength = 0.14f;
    float bloomThreshold = 1.0f;
    float aoStrength = 0.9f;
    float parallaxHeightScale = 0.018f;

    float globalRoughnessMultiplier = 1.0f;
    float globalMetallicMultiplier = 1.0f;

    bool enableShadows = true;
    bool enableBloom = true;
    bool enableFxaa = true;
    int shadowResolution = 2048;
    int shadowPcfRadius = 1;
  };

  bool Initialize();
  void Shutdown();

  void DrawScene(const std::vector<Component>& components, std::uint32_t selectedId, const OrbitCamera& camera, int viewportWidth, int viewportHeight, bool drawFloor);

  Settings& MutableSettings() { return settings_; }
  const Settings& GetSettings() const { return settings_; }

 private:
  bool BuildPrograms();
  bool InitializeFullscreenQuad();
  bool EnsureFramebuffers(int viewportWidth, int viewportHeight);
  bool EnsureShadowResources();
  void RenderSceneGeometry(GLuint program, const std::vector<Component>& components, std::uint32_t selectedId, bool drawFloor, bool drawSelectionOverlay);
  void DrawMeshByType(GeometryType type) const;

  GLuint pbrProgram_ = 0;
  GLuint depthProgram_ = 0;
  GLuint blurProgram_ = 0;
  GLuint postProgram_ = 0;

  GLuint hdrFbo_ = 0;
  GLuint hdrColorTex_ = 0;
  GLuint hdrBrightTex_ = 0;
  GLuint hdrDepthRbo_ = 0;
  GLuint pingPongFbo_[2] = {0, 0};
  GLuint pingPongTex_[2] = {0, 0};

  GLuint shadowFbo_ = 0;
  GLuint shadowDepthTex_ = 0;

  GLuint envCubemap_ = 0;
  GLuint brdfLutTex_ = 0;

  GLuint quadVao_ = 0;
  GLuint quadVbo_ = 0;

  int viewportWidth_ = 0;
  int viewportHeight_ = 0;

  Settings settings_;

  GpuMesh cuboid_;
  GpuMesh cylinder_;
  GpuMesh prism_;
  GpuMesh plane_;
};
