#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "bimmy_types.hpp"
#include "camera.hpp"
#include "raycast.hpp"
#include "renderer.hpp"

namespace {
constexpr int kWindowWidth = 1400;
constexpr int kWindowHeight = 860;

struct AppState {
  std::vector<Component> components;
  std::uint32_t nextId = 1;
  std::uint32_t selectedId = 0;
  OrbitCamera camera;
  float pendingScroll = 0.0f;
  bool rotatingCamera = false;
  double lastMouseX = 0.0;
  double lastMouseY = 0.0;
  bool freeLookEnabled = false;
  bool freeLookToggleHeld = false;
  bool sidebarVisible = true;
  float sidebarAnim = 1.0f;
  bool showFloor = false;

  enum class TransformMode {
    None,
    Translate,
    Rotate,
    Scale,
  };

  enum class GizmoAxis {
    None,
    X,
    Y,
    Z,
    XY,
    YZ,
    XZ,
  };

  TransformMode transformMode = TransformMode::None;
  GizmoAxis activeAxis = GizmoAxis::None;
  GizmoAxis constrainedAxis = GizmoAxis::None;
  bool draggingGizmo = false;

  bool gHeld = false;
  bool rHeld = false;
  bool sHeld = false;
  bool escHeld = false;
  bool xHeld = false;
  bool yHeld = false;
  bool zHeld = false;

  glm::vec2 dragStartMouse = glm::vec2(0.0f);
  glm::vec3 dragStartPosition = glm::vec3(0.0f);
  glm::vec3 dragStartRotation = glm::vec3(0.0f);
  glm::vec3 dragStartScale = glm::vec3(1.0f);
  float dragStartAngle = 0.0f;
  glm::vec3 dragAxisWorld = glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec3 dragAxisWorld2 = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec2 dragAxisScreenDir = glm::vec2(1.0f, 0.0f);
  glm::vec2 dragAxisScreenDir2 = glm::vec2(0.0f, 1.0f);
  float dragWorldPerPixel = 0.01f;
};

void UpdateFreeLookToggle(AppState& app, GLFWwindow* window, bool allowKeyboardInput) {
  if (!allowKeyboardInput) {
    app.freeLookToggleHeld = false;
    return;
  }

  const bool keyDown = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
  if (keyDown && !app.freeLookToggleHeld) {
    app.freeLookEnabled = !app.freeLookEnabled;
    app.rotatingCamera = false;

    if (app.freeLookEnabled) {
      app.camera.EnableFirstPerson();
      glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
      glfwGetCursorPos(window, &app.lastMouseX, &app.lastMouseY);
      app.rotatingCamera = true;
    } else {
      app.camera.DisableFirstPerson();
      glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
      app.rotatingCamera = false;
    }
  }
  app.freeLookToggleHeld = keyDown;
}

void SyncCursorForFreeLook(const AppState& app, GLFWwindow* window) {
  const int desired = app.freeLookEnabled ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL;
  if (glfwGetInputMode(window, GLFW_CURSOR) != desired) {
    glfwSetInputMode(window, GLFW_CURSOR, desired);
  }
}

void UpdateCameraKeyboard(AppState& app, GLFWwindow* window, float dt, bool allowKeyboardInput) {
  if (!allowKeyboardInput || dt <= 0.0f) {
    return;
  }

  // Keep camera keys available in orbit mode, but avoid fighting with modal gizmo tools.
  if (!app.freeLookEnabled && app.transformMode != AppState::TransformMode::None) {
    return;
  }

  float speed = 6.0f;
  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
    speed *= 2.5f;
  }

  glm::vec3 forward = app.camera.ForwardVector();
  forward.y = 0.0f;
  if (glm::length(forward) < 1e-4f) {
    const float yaw = glm::radians(app.camera.yawDeg);
    forward = glm::normalize(glm::vec3(cosf(yaw), 0.0f, sinf(yaw)));
  } else {
    forward = glm::normalize(forward);
  }

  const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
  const glm::vec3 up(0.0f, 1.0f, 0.0f);

  glm::vec3 delta(0.0f);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) delta += forward;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) delta -= forward;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) delta -= right;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) delta += right;
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) delta += up;
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) delta -= up;

  if (glm::length(delta) > 0.0f) {
    const glm::vec3 step = glm::normalize(delta) * speed * dt;
    if (app.camera.firstPerson) {
      app.camera.firstPersonPosition += step;
    } else {
      app.camera.target += step;
    }
  }
}

glm::vec3 AxisVector(AppState::GizmoAxis axis) {
  if (axis == AppState::GizmoAxis::X) return glm::vec3(1.0f, 0.0f, 0.0f);
  if (axis == AppState::GizmoAxis::Y) return glm::vec3(0.0f, 1.0f, 0.0f);
  return glm::vec3(0.0f, 0.0f, 1.0f);
}

bool ProjectWorldToScreen(const glm::vec3& world,
                          const glm::mat4& view,
                          const glm::mat4& proj,
                          int width,
                          int height,
                          glm::vec2& out) {
  if (width <= 0 || height <= 0) {
    return false;
  }

  const glm::vec4 clip = proj * view * glm::vec4(world, 1.0f);
  if (clip.w <= 0.0001f) {
    return false;
  }

  const glm::vec3 ndc = glm::vec3(clip) / clip.w;
  out.x = (ndc.x * 0.5f + 0.5f) * static_cast<float>(width);
  out.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(height);
  return true;
}

float DistanceToSegment(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b) {
  const glm::vec2 ab = b - a;
  const float denom = glm::dot(ab, ab);
  if (denom < 1e-5f) {
    return glm::length(p - a);
  }
  const float t = glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
  const glm::vec2 q = a + ab * t;
  return glm::length(p - q);
}

float DistanceToPolyline(const glm::vec2& p, const std::vector<glm::vec2>& polyline) {
  if (polyline.size() < 2) {
    return 1e9f;
  }

  float best = 1e9f;
  for (size_t i = 1; i < polyline.size(); ++i) {
    best = std::min(best, DistanceToSegment(p, polyline[i - 1], polyline[i]));
  }
  return best;
}

float WrapSignedRadians(float a) {
  constexpr float kTau = 6.28318530718f;
  while (a > 3.14159265359f) a -= kTau;
  while (a < -3.14159265359f) a += kTau;
  return a;
}

void ComputeLocalAxes(const Component& selected, glm::vec3& x, glm::vec3& y, glm::vec3& z) {
  glm::mat4 rot(1.0f);
  rot = glm::rotate(rot, glm::radians(selected.transform.rotationEulerDeg.x), glm::vec3(1, 0, 0));
  rot = glm::rotate(rot, glm::radians(selected.transform.rotationEulerDeg.y), glm::vec3(0, 1, 0));
  rot = glm::rotate(rot, glm::radians(selected.transform.rotationEulerDeg.z), glm::vec3(0, 0, 1));

  x = glm::normalize(glm::vec3(rot * glm::vec4(1, 0, 0, 0)));
  y = glm::normalize(glm::vec3(rot * glm::vec4(0, 1, 0, 0)));
  z = glm::normalize(glm::vec3(rot * glm::vec4(0, 0, 1, 0)));
}

bool PointInTriangle(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
  const glm::vec2 v0 = c - a;
  const glm::vec2 v1 = b - a;
  const glm::vec2 v2 = p - a;

  const float dot00 = glm::dot(v0, v0);
  const float dot01 = glm::dot(v0, v1);
  const float dot02 = glm::dot(v0, v2);
  const float dot11 = glm::dot(v1, v1);
  const float dot12 = glm::dot(v1, v2);

  const float denom = dot00 * dot11 - dot01 * dot01;
  if (fabsf(denom) < 1e-6f) {
    return false;
  }

  const float inv = 1.0f / denom;
  const float u = (dot11 * dot02 - dot01 * dot12) * inv;
  const float v = (dot00 * dot12 - dot01 * dot02) * inv;
  return u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f;
}

bool PointInQuad(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d) {
  return PointInTriangle(p, a, b, c) || PointInTriangle(p, a, c, d);
}

void HandleTransformModeShortcuts(AppState& app, GLFWwindow* window, bool allowKeyboardInput) {
  if (!allowKeyboardInput) {
    app.gHeld = app.rHeld = app.sHeld = app.escHeld = false;
    app.xHeld = app.yHeld = app.zHeld = false;
    return;
  }

  const bool gDown = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
  const bool rDown = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
  const bool sDown = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
  const bool escDown = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
  const bool xDown = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
  const bool yDown = glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS;
  const bool zDown = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;

  if (gDown && !app.gHeld) {
    app.transformMode = (app.transformMode == AppState::TransformMode::Translate) ? AppState::TransformMode::None : AppState::TransformMode::Translate;
    app.draggingGizmo = false;
    app.activeAxis = AppState::GizmoAxis::None;
    app.constrainedAxis = AppState::GizmoAxis::None;
  }
  if (rDown && !app.rHeld) {
    app.transformMode = (app.transformMode == AppState::TransformMode::Rotate) ? AppState::TransformMode::None : AppState::TransformMode::Rotate;
    app.draggingGizmo = false;
    app.activeAxis = AppState::GizmoAxis::None;
    app.constrainedAxis = AppState::GizmoAxis::None;
  }
  if (sDown && !app.sHeld) {
    app.transformMode = (app.transformMode == AppState::TransformMode::Scale) ? AppState::TransformMode::None : AppState::TransformMode::Scale;
    app.draggingGizmo = false;
    app.activeAxis = AppState::GizmoAxis::None;
    app.constrainedAxis = AppState::GizmoAxis::None;
  }
  if (escDown && !app.escHeld) {
    app.transformMode = AppState::TransformMode::None;
    app.draggingGizmo = false;
    app.activeAxis = AppState::GizmoAxis::None;
    app.constrainedAxis = AppState::GizmoAxis::None;
  }

  if (app.transformMode != AppState::TransformMode::None) {
    if (xDown && !app.xHeld) {
      app.constrainedAxis = (app.constrainedAxis == AppState::GizmoAxis::X) ? AppState::GizmoAxis::None : AppState::GizmoAxis::X;
    }
    if (yDown && !app.yHeld) {
      app.constrainedAxis = (app.constrainedAxis == AppState::GizmoAxis::Y) ? AppState::GizmoAxis::None : AppState::GizmoAxis::Y;
    }
    if (zDown && !app.zHeld) {
      app.constrainedAxis = (app.constrainedAxis == AppState::GizmoAxis::Z) ? AppState::GizmoAxis::None : AppState::GizmoAxis::Z;
    }
  }

  app.gHeld = gDown;
  app.rHeld = rDown;
  app.sHeld = sDown;
  app.escHeld = escDown;
  app.xHeld = xDown;
  app.yHeld = yDown;
  app.zHeld = zDown;
}

bool UpdateTransformGizmo(AppState& app,
                          Component& selected,
                          const OrbitCamera& camera,
                          int fbW,
                          int fbH,
                          bool allowSceneMouse,
                          bool leftDown,
                          bool leftPressed,
                          bool leftReleased,
                          double mouseX,
                          double mouseY) {
  if (app.transformMode == AppState::TransformMode::None || fbW <= 0 || fbH <= 0) {
    return false;
  }

  const glm::vec3 centerWorld = selected.transform.position;
  const glm::mat4 view = camera.ViewMatrix();
  const glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(fbW) / static_cast<float>(fbH), 0.1f, 500.0f);

  glm::vec2 centerScreen;
  if (!ProjectWorldToScreen(centerWorld, view, proj, fbW, fbH, centerScreen)) {
    app.draggingGizmo = false;
    app.activeAxis = AppState::GizmoAxis::None;
    return false;
  }

  const float dist = glm::length(camera.Position() - centerWorld);
  const float gizmoScale = glm::max(0.6f, dist * 0.18f);
  const float planeScale = gizmoScale * 0.45f;

  glm::vec3 localX, localY, localZ;
  ComputeLocalAxes(selected, localX, localY, localZ);

  const glm::vec2 mouse(static_cast<float>(mouseX), static_cast<float>(mouseY));

  struct AxisDrawData {
    AppState::GizmoAxis axis;
    glm::vec3 axisWorld;
    glm::vec2 endScreen;
    bool visible;
  };

  AxisDrawData axisData[3] = {
    {AppState::GizmoAxis::X, localX, glm::vec2(0.0f), false},
    {AppState::GizmoAxis::Y, localY, glm::vec2(0.0f), false},
    {AppState::GizmoAxis::Z, localZ, glm::vec2(0.0f), false},
  };

  struct PlaneDrawData {
    AppState::GizmoAxis plane;
    glm::vec3 axisA;
    glm::vec3 axisB;
    glm::vec2 s0;
    glm::vec2 s1;
    glm::vec2 s2;
    glm::vec2 s3;
    bool visible = false;
  };

  PlaneDrawData planeData[3] = {
    {AppState::GizmoAxis::XY, localX, localY},
    {AppState::GizmoAxis::YZ, localY, localZ},
    {AppState::GizmoAxis::XZ, localX, localZ},
  };

  ImDrawList* draw = ImGui::GetForegroundDrawList();
  const ImU32 xColor = IM_COL32(255, 90, 90, 235);
  const ImU32 yColor = IM_COL32(95, 235, 125, 235);
  const ImU32 zColor = IM_COL32(90, 130, 255, 235);
  const ImU32 activeColor = IM_COL32(255, 235, 120, 255);

  auto axisColor = [&](AppState::GizmoAxis axis) {
    if (axis == AppState::GizmoAxis::X) return xColor;
    if (axis == AppState::GizmoAxis::Y) return yColor;
    return zColor;
  };

  auto axisFromId = [&](AppState::GizmoAxis axisId) {
    if (axisId == AppState::GizmoAxis::X) return localX;
    if (axisId == AppState::GizmoAxis::Y) return localY;
    return localZ;
  };

  auto computeAxisScreenDir = [&](const glm::vec3& axisWorld, glm::vec2& outDir) {
    glm::vec2 axisEnd;
    if (ProjectWorldToScreen(centerWorld + axisWorld * gizmoScale, view, proj, fbW, fbH, axisEnd)) {
      glm::vec2 d = axisEnd - centerScreen;
      const float len = glm::length(d);
      if (len > 1e-4f) {
        outDir = d / len;
        return;
      }
    }
    outDir = glm::vec2(1.0f, 0.0f);
  };

  if (app.transformMode == AppState::TransformMode::Rotate) {
    struct RingData {
      AppState::GizmoAxis axis;
      std::vector<glm::vec2> screenPoints;
    };

    auto BuildRing = [&](AppState::GizmoAxis axis) {
      RingData ring;
      ring.axis = axis;

      glm::vec3 u(1.0f, 0.0f, 0.0f);
      glm::vec3 v(0.0f, 1.0f, 0.0f);
      if (axis == AppState::GizmoAxis::X) {
        u = localY;
        v = localZ;
      } else if (axis == AppState::GizmoAxis::Y) {
        u = localX;
        v = localZ;
      } else {
        u = localX;
        v = localY;
      }

      constexpr int kSegments = 96;
      constexpr float kTau = 6.28318530718f;
      ring.screenPoints.reserve(kSegments + 1);
      for (int i = 0; i <= kSegments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSegments);
        const float ang = t * kTau;
        const glm::vec3 world = centerWorld + (u * cosf(ang) + v * sinf(ang)) * gizmoScale;
        glm::vec2 screen;
        if (ProjectWorldToScreen(world, view, proj, fbW, fbH, screen)) {
          ring.screenPoints.push_back(screen);
        }
      }
      return ring;
    };

    RingData rings[3] = {
      BuildRing(AppState::GizmoAxis::X),
      BuildRing(AppState::GizmoAxis::Y),
      BuildRing(AppState::GizmoAxis::Z),
    };

    AppState::GizmoAxis hoveredAxis = AppState::GizmoAxis::None;
    if (app.constrainedAxis == AppState::GizmoAxis::X ||
        app.constrainedAxis == AppState::GizmoAxis::Y ||
        app.constrainedAxis == AppState::GizmoAxis::Z) {
      hoveredAxis = app.constrainedAxis;
    } else {
      float bestDist = 10.0f;
      for (const RingData& ring : rings) {
        const float d = DistanceToPolyline(mouse, ring.screenPoints);
        if (d < bestDist) {
          bestDist = d;
          hoveredAxis = ring.axis;
        }
      }
    }

    if (allowSceneMouse && leftPressed && hoveredAxis != AppState::GizmoAxis::None) {
      app.draggingGizmo = true;
      app.activeAxis = hoveredAxis;
      app.dragStartRotation = selected.transform.rotationEulerDeg;
      app.dragStartAngle = atan2f(mouse.y - centerScreen.y, mouse.x - centerScreen.x);
      app.dragAxisWorld = axisFromId(hoveredAxis);
    }

    if (app.draggingGizmo) {
      if (leftDown) {
        const float currentAngle = atan2f(mouse.y - centerScreen.y, mouse.x - centerScreen.x);
        const float deltaDeg = glm::degrees(WrapSignedRadians(currentAngle - app.dragStartAngle));
        selected.transform.rotationEulerDeg = app.dragStartRotation;
        if (app.activeAxis == AppState::GizmoAxis::X) selected.transform.rotationEulerDeg.x += deltaDeg;
        if (app.activeAxis == AppState::GizmoAxis::Y) selected.transform.rotationEulerDeg.y += deltaDeg;
        if (app.activeAxis == AppState::GizmoAxis::Z) selected.transform.rotationEulerDeg.z += deltaDeg;
      }

      if (leftReleased) {
        app.draggingGizmo = false;
        app.activeAxis = AppState::GizmoAxis::None;
      }
    }

    for (const RingData& ring : rings) {
      if (ring.screenPoints.size() < 2) {
        continue;
      }

      ImU32 color = axisColor(ring.axis);
      if (ring.axis == hoveredAxis || ring.axis == app.activeAxis) {
        color = activeColor;
      }

      for (size_t i = 1; i < ring.screenPoints.size(); ++i) {
        const glm::vec2& a = ring.screenPoints[i - 1];
        const glm::vec2& b = ring.screenPoints[i];
        draw->AddLine(ImVec2(a.x, a.y), ImVec2(b.x, b.y), color, 2.2f);
      }
    }

    draw->AddCircleFilled(ImVec2(centerScreen.x, centerScreen.y), 3.0f, IM_COL32(240, 240, 240, 200));
    return app.draggingGizmo || hoveredAxis != AppState::GizmoAxis::None;
  }

  for (AxisDrawData& axis : axisData) {
    glm::vec2 end;
    axis.visible = ProjectWorldToScreen(centerWorld + axis.axisWorld * gizmoScale, view, proj, fbW, fbH, end);
    axis.endScreen = end;
  }

  AppState::GizmoAxis hoveredAxis = AppState::GizmoAxis::None;
  if (app.constrainedAxis == AppState::GizmoAxis::X ||
      app.constrainedAxis == AppState::GizmoAxis::Y ||
      app.constrainedAxis == AppState::GizmoAxis::Z) {
    hoveredAxis = app.constrainedAxis;
  } else {
    float bestDist = 12.0f;
    for (const AxisDrawData& axis : axisData) {
      if (!axis.visible) continue;
      const float d = DistanceToSegment(mouse, centerScreen, axis.endScreen);
      if (d < bestDist) {
        bestDist = d;
        hoveredAxis = axis.axis;
      }
    }

    if (app.transformMode != AppState::TransformMode::Rotate) {
      for (PlaneDrawData& plane : planeData) {
        glm::vec2 s0, s1, s2, s3;
        plane.visible = ProjectWorldToScreen(centerWorld + (plane.axisA + plane.axisB) * (planeScale * 0.15f), view, proj, fbW, fbH, s0) &&
                        ProjectWorldToScreen(centerWorld + (plane.axisA * 0.85f + plane.axisB * 0.15f) * planeScale, view, proj, fbW, fbH, s1) &&
                        ProjectWorldToScreen(centerWorld + (plane.axisA + plane.axisB) * (planeScale * 0.85f), view, proj, fbW, fbH, s2) &&
                        ProjectWorldToScreen(centerWorld + (plane.axisA * 0.15f + plane.axisB * 0.85f) * planeScale, view, proj, fbW, fbH, s3);
        plane.s0 = s0;
        plane.s1 = s1;
        plane.s2 = s2;
        plane.s3 = s3;

        if (plane.visible && PointInQuad(mouse, s0, s1, s2, s3)) {
          hoveredAxis = plane.plane;
        }
      }
    }
  }

  if (allowSceneMouse && leftPressed && hoveredAxis != AppState::GizmoAxis::None) {
    app.draggingGizmo = true;
    app.activeAxis = hoveredAxis;
    app.dragStartMouse = mouse;
    app.dragStartPosition = selected.transform.position;
    app.dragStartRotation = selected.transform.rotationEulerDeg;
    app.dragStartScale = selected.transform.scale;
    app.dragWorldPerPixel = glm::max(0.002f, dist * 0.0024f);

    if (hoveredAxis == AppState::GizmoAxis::XY || hoveredAxis == AppState::GizmoAxis::YZ || hoveredAxis == AppState::GizmoAxis::XZ) {
      if (hoveredAxis == AppState::GizmoAxis::XY) {
        app.dragAxisWorld = localX;
        app.dragAxisWorld2 = localY;
      } else if (hoveredAxis == AppState::GizmoAxis::YZ) {
        app.dragAxisWorld = localY;
        app.dragAxisWorld2 = localZ;
      } else {
        app.dragAxisWorld = localX;
        app.dragAxisWorld2 = localZ;
      }

      computeAxisScreenDir(app.dragAxisWorld, app.dragAxisScreenDir);
      computeAxisScreenDir(app.dragAxisWorld2, app.dragAxisScreenDir2);
    } else {
      app.dragAxisWorld = axisFromId(hoveredAxis);
      app.dragAxisWorld2 = glm::vec3(0.0f);
      computeAxisScreenDir(app.dragAxisWorld, app.dragAxisScreenDir);
      app.dragAxisScreenDir2 = glm::vec2(0.0f);
    }
  }

  if (app.draggingGizmo) {
    if (leftDown) {
      const glm::vec2 dragDelta = mouse - app.dragStartMouse;
      const float axisPixels = glm::dot(dragDelta, app.dragAxisScreenDir);
      const float axisPixels2 = glm::dot(dragDelta, app.dragAxisScreenDir2);
      const bool planeDrag = app.activeAxis == AppState::GizmoAxis::XY || app.activeAxis == AppState::GizmoAxis::YZ || app.activeAxis == AppState::GizmoAxis::XZ;

      if (app.transformMode == AppState::TransformMode::Translate) {
        if (planeDrag) {
          selected.transform.position = app.dragStartPosition +
            app.dragAxisWorld * (axisPixels * app.dragWorldPerPixel) +
            app.dragAxisWorld2 * (axisPixels2 * app.dragWorldPerPixel);
        } else {
          selected.transform.position = app.dragStartPosition + app.dragAxisWorld * (axisPixels * app.dragWorldPerPixel);
        }
      } else if (app.transformMode == AppState::TransformMode::Rotate) {
        const float degrees = axisPixels * 0.35f;
        selected.transform.rotationEulerDeg = app.dragStartRotation;
        if (app.activeAxis == AppState::GizmoAxis::X) selected.transform.rotationEulerDeg.x += degrees;
        if (app.activeAxis == AppState::GizmoAxis::Y) selected.transform.rotationEulerDeg.y += degrees;
        if (app.activeAxis == AppState::GizmoAxis::Z) selected.transform.rotationEulerDeg.z += degrees;
      } else if (app.transformMode == AppState::TransformMode::Scale) {
        const float scaleAmount = axisPixels * 0.006f;
        const float scaleAmount2 = axisPixels2 * 0.006f;
        selected.transform.scale = app.dragStartScale;
        if (app.activeAxis == AppState::GizmoAxis::X) selected.transform.scale.x += scaleAmount;
        if (app.activeAxis == AppState::GizmoAxis::Y) selected.transform.scale.y += scaleAmount;
        if (app.activeAxis == AppState::GizmoAxis::Z) selected.transform.scale.z += scaleAmount;
        if (app.activeAxis == AppState::GizmoAxis::XY) {
          selected.transform.scale.x += scaleAmount;
          selected.transform.scale.y += scaleAmount2;
        }
        if (app.activeAxis == AppState::GizmoAxis::YZ) {
          selected.transform.scale.y += scaleAmount;
          selected.transform.scale.z += scaleAmount2;
        }
        if (app.activeAxis == AppState::GizmoAxis::XZ) {
          selected.transform.scale.x += scaleAmount;
          selected.transform.scale.z += scaleAmount2;
        }
        selected.transform.scale = glm::max(selected.transform.scale, glm::vec3(0.05f));
      }
    }

    if (leftReleased) {
      app.draggingGizmo = false;
      app.activeAxis = AppState::GizmoAxis::None;
    }
  }

  for (const AxisDrawData& axis : axisData) {
    if (!axis.visible) continue;

    ImU32 color = zColor;
    if (axis.axis == AppState::GizmoAxis::X) color = xColor;
    if (axis.axis == AppState::GizmoAxis::Y) color = yColor;
    if (axis.axis == hoveredAxis || axis.axis == app.activeAxis) color = activeColor;

    draw->AddLine(ImVec2(centerScreen.x, centerScreen.y), ImVec2(axis.endScreen.x, axis.endScreen.y), color, 3.0f);

    if (app.transformMode == AppState::TransformMode::Translate) {
      draw->AddCircleFilled(ImVec2(axis.endScreen.x, axis.endScreen.y), 4.5f, color);
    } else if (app.transformMode == AppState::TransformMode::Scale) {
      draw->AddRectFilled(ImVec2(axis.endScreen.x - 4.0f, axis.endScreen.y - 4.0f), ImVec2(axis.endScreen.x + 4.0f, axis.endScreen.y + 4.0f), color);
    } else {
      draw->AddCircle(ImVec2(axis.endScreen.x, axis.endScreen.y), 5.0f, color, 18, 2.0f);
    }
  }

  if (app.transformMode == AppState::TransformMode::Translate || app.transformMode == AppState::TransformMode::Scale) {
    for (const PlaneDrawData& plane : planeData) {
      if (!plane.visible) continue;

      ImU32 baseColor = IM_COL32(200, 200, 200, 56);
      if (plane.plane == AppState::GizmoAxis::XY) baseColor = IM_COL32(255, 255, 90, 68);
      if (plane.plane == AppState::GizmoAxis::YZ) baseColor = IM_COL32(90, 255, 220, 68);
      if (plane.plane == AppState::GizmoAxis::XZ) baseColor = IM_COL32(255, 130, 255, 68);
      if (plane.plane == hoveredAxis || plane.plane == app.activeAxis) baseColor = IM_COL32(255, 235, 120, 110);

      draw->AddQuadFilled(ImVec2(plane.s0.x, plane.s0.y), ImVec2(plane.s1.x, plane.s1.y), ImVec2(plane.s2.x, plane.s2.y), ImVec2(plane.s3.x, plane.s3.y), baseColor);
      draw->AddQuad(ImVec2(plane.s0.x, plane.s0.y), ImVec2(plane.s1.x, plane.s1.y), ImVec2(plane.s2.x, plane.s2.y), ImVec2(plane.s3.x, plane.s3.y), IM_COL32(240, 240, 240, 130), 1.0f);
    }
  }

  draw->AddCircleFilled(ImVec2(centerScreen.x, centerScreen.y), 4.0f, IM_COL32(240, 240, 240, 220));
  return app.draggingGizmo || hoveredAxis != AppState::GizmoAxis::None;
}

Component* FindById(std::vector<Component>& components, std::uint32_t id) {
  for (Component& c : components) {
    if (c.id == id) return &c;
  }
  return nullptr;
}

void AddDefaultComponent(AppState& app, GeometryType type) {
  Component c;
  c.id = app.nextId++;
  c.geometry = type;
  c.material = MaterialType::Concrete;
  c.transform.position = glm::vec3(0.0f, 1.0f, 0.0f);

  if (type == GeometryType::Cuboid) {
    c.dimensions = glm::vec3(2.5f, 1.2f, 0.4f);
  } else if (type == GeometryType::Cylinder) {
    c.dimensions = glm::vec3(0.35f, 2.6f, 0.35f);
  } else {
    c.dimensions = glm::vec3(2.0f, 1.4f, 1.2f);
  }

  app.components.push_back(c);
  app.selectedId = c.id;
}

constexpr std::array<MaterialType, 7> kMaterialOrder = {
  MaterialType::SheetMetal,
  MaterialType::Grass,
  MaterialType::Concrete,
  MaterialType::RustedMetal,
  MaterialType::Brick,
  MaterialType::Roof,
  MaterialType::Wood,
};

int MaterialIndex(MaterialType type) {
  for (int i = 0; i < static_cast<int>(kMaterialOrder.size()); ++i) {
    if (kMaterialOrder[i] == type) {
      return i;
    }
  }
  return 0;
}

MaterialType MaterialFromIndex(int index) {
  if (index < 0 || index >= static_cast<int>(kMaterialOrder.size())) {
    return kMaterialOrder[0];
  }
  return kMaterialOrder[static_cast<std::size_t>(index)];
}

void DrawUi(AppState& app, float fps) {
  ImGuiIO& io = ImGui::GetIO();
  ImGuiViewport* viewport = ImGui::GetMainViewport();

  const float inspectorWidth = 380.0f;
  const float targetAnim = app.sidebarVisible ? 1.0f : 0.0f;
  const float blend = std::min(1.0f, io.DeltaTime * 12.0f);
  app.sidebarAnim += (targetAnim - app.sidebarAnim) * blend;
  const float currentSidebarWidth = inspectorWidth * app.sidebarAnim;

  if (currentSidebarWidth > 4.0f) {
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - currentSidebarWidth, viewport->WorkPos.y));
    ImGui::SetNextWindowSize(ImVec2(currentSidebarWidth, viewport->WorkSize.y));
    ImGui::SetNextWindowBgAlpha(0.32f);
    ImGui::Begin("Inspector & Inventory", nullptr,
      ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoTitleBar);

    ImGui::Text("BIMMY MVP");
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Separator();
    ImGui::Text("Free-look: %s (toggle: F)", app.freeLookEnabled ? "ON" : "OFF");
    ImGui::Checkbox("Show Floor Plane", &app.showFloor);

    if (ImGui::Button("Add Cuboid")) AddDefaultComponent(app, GeometryType::Cuboid);
    ImGui::SameLine();
    if (ImGui::Button("Add Cylinder")) AddDefaultComponent(app, GeometryType::Cylinder);
    ImGui::SameLine();
    if (ImGui::Button("Add Prism")) AddDefaultComponent(app, GeometryType::Prism);

    Component* selected = FindById(app.components, app.selectedId);
    ImGui::SeparatorText("Inspector");
    if (selected == nullptr) {
      ImGui::Text("No component selected.");
    } else {
      ImGui::Text("Component ID: %u", selected->id);

      const char* shapeLabel = "Cuboid";
      if (selected->geometry == GeometryType::Cylinder) shapeLabel = "Cylinder";
      if (selected->geometry == GeometryType::Prism) shapeLabel = "Prism";
      ImGui::Text("Geometry: %s", shapeLabel);

      if (selected->geometry == GeometryType::Cylinder) {
        ImGui::DragFloat("Radius", &selected->dimensions.x, 0.02f, 0.05f, 20.0f);
        ImGui::DragFloat("Height", &selected->dimensions.y, 0.04f, 0.05f, 40.0f);
        selected->dimensions.z = selected->dimensions.x;
      } else {
        ImGui::DragFloat3("Dimensions", &selected->dimensions.x, 0.04f, 0.05f, 60.0f);
      }

      ImGui::DragFloat3("Position", &selected->transform.position.x, 0.05f);
      ImGui::DragFloat3("Rotation", &selected->transform.rotationEulerDeg.x, 0.7f);
      ImGui::DragFloat3("Scale", &selected->transform.scale.x, 0.02f, 0.05f, 30.0f);

      int materialIndex = MaterialIndex(selected->material);
      const auto& catalog = MaterialCatalog();

      if (ImGui::BeginCombo("Material", catalog.at(MaterialFromIndex(materialIndex)).name.c_str())) {
        for (int i = 0; i < static_cast<int>(kMaterialOrder.size()); ++i) {
          const MaterialType type = MaterialFromIndex(i);
          const bool isSelected = (selected->material == type);
          if (ImGui::Selectable(catalog.at(type).name.c_str(), isSelected)) {
            selected->material = type;
          }
          if (isSelected) {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }

      ImGui::Text("Volume: %.3f", selected->Volume());
      ImGui::Text("Cost: $%.2f", selected->Cost());
    }

    BomTotals totals = ComputeBom(app.components);

    ImGui::SeparatorText("Inventory Dashboard");
    if (ImGui::BeginTable("bom_table", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Material");
      ImGui::TableSetupColumn("Volume");
      ImGui::TableSetupColumn("Cost");
      ImGui::TableHeadersRow();

      for (MaterialType m : kMaterialOrder) {
        const auto& def = MaterialCatalog().at(m);
        float v = 0.0f;
        float c = 0.0f;
        if (totals.volumeByMaterial.count(m)) v = totals.volumeByMaterial[m];
        if (totals.costByMaterial.count(m)) c = totals.costByMaterial[m];

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(def.name.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", v);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("$%.2f", c);
      }

      ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::Text("Grand Total: $%.2f", totals.grandTotal);
    ImGui::TextDisabled("Orbit: RMB drag, Zoom: Wheel, Pick: Left Click");
    ImGui::TextDisabled("Free-look: press F to toggle mouse-look");
    ImGui::TextDisabled("Transforms: G/R/S, drag axis or plane handles (XY/YZ/XZ), X/Y/Z constrain, Esc cancel");

    ImGui::End();
  }

  const float toggleW = 44.0f;
  const float toggleH = 36.0f;
  const float togglePad = 8.0f;
  float toggleX = viewport->WorkPos.x + viewport->WorkSize.x - toggleW - togglePad;
  if (currentSidebarWidth > 4.0f) {
    toggleX = viewport->WorkPos.x + viewport->WorkSize.x - currentSidebarWidth - toggleW - togglePad;
  }

  ImGui::SetNextWindowPos(ImVec2(toggleX, viewport->WorkPos.y + 10.0f));
  ImGui::SetNextWindowSize(ImVec2(toggleW, toggleH));
  ImGui::SetNextWindowBgAlpha(0.22f);
  ImGui::Begin("##SidebarToggle", nullptr,
    ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoCollapse |
    ImGuiWindowFlags_NoTitleBar |
    ImGuiWindowFlags_NoScrollbar |
    ImGuiWindowFlags_NoNav |
    ImGuiWindowFlags_NoFocusOnAppearing);
  if (ImGui::Button(app.sidebarVisible ? "<" : ">", ImVec2(-1.0f, -1.0f))) {
    app.sidebarVisible = !app.sidebarVisible;
  }
  ImGui::End();

  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y));
  ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y));
  ImGui::Begin("Viewport", nullptr,
    ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoCollapse |
    ImGuiWindowFlags_NoBackground |
    ImGuiWindowFlags_NoBringToFrontOnFocus |
    ImGuiWindowFlags_NoInputs |
    ImGuiWindowFlags_NoNav);

  ImGui::Text("3D Scene");
  ImGui::TextDisabled("Components: %zu", app.components.size());
  ImGui::TextDisabled("Camera: RMB orbit, or F for FPS free-look (WASD/QE + mouse)");
  ImGui::End();

  if (app.freeLookEnabled) {
    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    const ImVec2 center(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f, viewport->WorkPos.y + viewport->WorkSize.y * 0.5f);
    const float arm = 8.0f;
    const float gap = 4.0f;
    const ImU32 color = IM_COL32(255, 235, 160, 230);

    drawList->AddLine(ImVec2(center.x - arm, center.y), ImVec2(center.x - gap, center.y), color, 2.0f);
    drawList->AddLine(ImVec2(center.x + gap, center.y), ImVec2(center.x + arm, center.y), color, 2.0f);
    drawList->AddLine(ImVec2(center.x, center.y - arm), ImVec2(center.x, center.y - gap), color, 2.0f);
    drawList->AddLine(ImVec2(center.x, center.y + gap), ImVec2(center.x, center.y + arm), color, 2.0f);
  }

  (void)io;
}
}

int main() {
  if (!glfwInit()) {
    std::fprintf(stderr, "Failed to initialize GLFW\n");
    return EXIT_FAILURE;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "BIMMY - OpenGL MVP", nullptr, nullptr);
  if (window == nullptr) {
    std::fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return EXIT_FAILURE;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  if (!gladLoadGL(glfwGetProcAddress)) {
    std::fprintf(stderr, "Failed to initialize GLAD\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
  (void)io;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  Renderer renderer;
  if (!renderer.Initialize()) {
    std::fprintf(stderr, "Failed to initialize renderer\n");
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  AppState app;
  glfwSetWindowUserPointer(window, &app);
  glfwSetScrollCallback(window, [](GLFWwindow* w, double, double yoff) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(w));
    if (state != nullptr) {
      state->pendingScroll += static_cast<float>(yoff);
    }
  });

  AddDefaultComponent(app, GeometryType::Cuboid);
  app.components.back().dimensions = glm::vec3(4.0f, 2.8f, 0.25f);
  app.components.back().transform.position = glm::vec3(0.0f, 1.4f, 0.0f);

  double lastTime = glfwGetTime();
  float fps = 0.0f;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const double now = glfwGetTime();
    const float dt = static_cast<float>(now - lastTime);
    lastTime = now;
    if (dt > 0.0f) fps = 1.0f / dt;

    int fbW = 0;
    int fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    DrawUi(app, fps);

    const bool allowSceneMouse = app.freeLookEnabled || !ImGui::GetIO().WantCaptureMouse;
    const bool allowSceneKeyboard = app.freeLookEnabled || !ImGui::GetIO().WantCaptureKeyboard;

    UpdateFreeLookToggle(app, window, allowSceneKeyboard);
    SyncCursorForFreeLook(app, window);

    UpdateCameraKeyboard(app, window, dt, allowSceneKeyboard);

    if (!app.freeLookEnabled && app.selectedId != 0) {
      HandleTransformModeShortcuts(app, window, allowSceneKeyboard);
    } else {
      app.transformMode = AppState::TransformMode::None;
      app.draggingGizmo = false;
      app.activeAxis = AppState::GizmoAxis::None;
    }

    double mx = 0.0;
    double my = 0.0;
    glfwGetCursorPos(window, &mx, &my);

    static bool wasLeftDown = false;
    bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool leftPressed = leftDown && !wasLeftDown;
    const bool leftReleased = !leftDown && wasLeftDown;

    bool gizmoCapturedMouse = false;
    if (!app.freeLookEnabled) {
      if (Component* selected = FindById(app.components, app.selectedId); selected != nullptr) {
        gizmoCapturedMouse = UpdateTransformGizmo(app, *selected, app.camera, fbW, fbH, allowSceneMouse, leftDown, leftPressed, leftReleased, mx, my);
      }
    }

    if (app.freeLookEnabled) {
      double mx = 0.0;
      double my = 0.0;
      glfwGetCursorPos(window, &mx, &my);

      if (!app.rotatingCamera) {
        app.rotatingCamera = true;
        app.lastMouseX = mx;
        app.lastMouseY = my;
      } else {
        const float dx = static_cast<float>(mx - app.lastMouseX);
        const float dy = static_cast<float>(my - app.lastMouseY);
        app.camera.Orbit(dx * 0.2f, -dy * 0.2f);
        app.lastMouseX = mx;
        app.lastMouseY = my;
      }
    }

    if (allowSceneMouse) {

      if (!app.freeLookEnabled) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
          if (!app.rotatingCamera) {
            app.rotatingCamera = true;
            app.lastMouseX = mx;
            app.lastMouseY = my;
          } else {
            const float dx = static_cast<float>(mx - app.lastMouseX);
            const float dy = static_cast<float>(my - app.lastMouseY);
            app.camera.Orbit(dx * 0.2f, dy * 0.2f);
            app.lastMouseX = mx;
            app.lastMouseY = my;
          }
        } else {
          app.rotatingCamera = false;
        }
      }

      if (leftPressed && !gizmoCapturedMouse) {
        if (fbW > 0 && fbH > 0) {
          double pickX = mx;
          double pickY = my;
          if (app.freeLookEnabled) {
            pickX = static_cast<double>(fbW) * 0.5;
            pickY = static_cast<double>(fbH) * 0.5;
          }

          const glm::mat4 view = app.camera.ViewMatrix();
          const glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(fbW) / static_cast<float>(fbH), 0.1f, 500.0f);
          Ray ray = BuildRayFromScreen(pickX, pickY, fbW, fbH, view, proj);
          auto picked = PickComponent(ray, app.components);
          app.selectedId = picked.value_or(0);
        }
      }
    }

    wasLeftDown = leftDown;

    if (allowSceneMouse && app.pendingScroll != 0.0f) {
      app.camera.Zoom(app.pendingScroll);
    }
    app.pendingScroll = 0.0f;

    renderer.DrawScene(app.components, app.selectedId, app.camera, fbW, fbH, app.showFloor);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  renderer.Shutdown();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
