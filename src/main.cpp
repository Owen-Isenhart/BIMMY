#include <algorithm>
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

void UpdateCameraKeyboard(AppState& app, GLFWwindow* window, float dt, bool allowKeyboardInput) {
  if (!allowKeyboardInput || dt <= 0.0f) {
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

void UpdateSelectedTransformKeyboard(Component& selected, GLFWwindow* window, float dt, bool allowKeyboardInput) {
  if (!allowKeyboardInput || dt <= 0.0f) {
    return;
  }

  const bool altPressed = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
  if (!altPressed) {
    return;
  }

  const float moveSpeed = 2.2f;
  const float rotateSpeed = 70.0f;
  const float scaleSpeed = 0.8f;

  if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) selected.transform.position.z -= moveSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) selected.transform.position.z += moveSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) selected.transform.position.x -= moveSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) selected.transform.position.x += moveSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS) selected.transform.position.y += moveSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) selected.transform.position.y -= moveSpeed * dt;

  if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) selected.transform.rotationEulerDeg.x += rotateSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) selected.transform.rotationEulerDeg.x -= rotateSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) selected.transform.rotationEulerDeg.y -= rotateSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) selected.transform.rotationEulerDeg.y += rotateSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) selected.transform.rotationEulerDeg.z -= rotateSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) selected.transform.rotationEulerDeg.z += rotateSpeed * dt;

  float scaleDelta = 0.0f;
  if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) scaleDelta -= scaleSpeed * dt;
  if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) scaleDelta += scaleSpeed * dt;
  if (scaleDelta != 0.0f) {
    selected.transform.scale += glm::vec3(scaleDelta);
    selected.transform.scale = glm::max(selected.transform.scale, glm::vec3(0.05f));
  }
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

void DrawUi(AppState& app, float fps) {
  ImGuiIO& io = ImGui::GetIO();
  ImGuiViewport* viewport = ImGui::GetMainViewport();

  const float inspectorWidth = 380.0f;

  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - inspectorWidth, viewport->WorkPos.y));
  ImGui::SetNextWindowSize(ImVec2(inspectorWidth, viewport->WorkSize.y));
  ImGui::Begin("Inspector & Inventory", nullptr,
    ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoCollapse);

  ImGui::Text("BIMMY MVP");
  ImGui::Text("FPS: %.1f", fps);
  ImGui::Separator();
  ImGui::Text("Free-look: %s (toggle: F)", app.freeLookEnabled ? "ON" : "OFF");

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

    int materialIndex = 0;
    if (selected->material == MaterialType::Steel) materialIndex = 1;
    if (selected->material == MaterialType::Timber) materialIndex = 2;

    if (ImGui::Combo("Material", &materialIndex, "Concrete\0Steel\0Timber\0")) {
      if (materialIndex == 0) selected->material = MaterialType::Concrete;
      if (materialIndex == 1) selected->material = MaterialType::Steel;
      if (materialIndex == 2) selected->material = MaterialType::Timber;
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

    for (MaterialType m : {MaterialType::Concrete, MaterialType::Steel, MaterialType::Timber}) {
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
  ImGui::TextDisabled("Component hotkeys: hold Alt + Arrows/PgUp/PgDn move, IJKLUO rotate, [ ] scale");

  ImGui::End();

  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y));
  ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x - inspectorWidth, viewport->WorkSize.y));
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
  ImGui::TextDisabled("Camera: RMB orbit or FPS free-look, wheel zoom, WASD move, Q/E down/up, Shift fast");
  ImGui::End();

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

    UpdateCameraKeyboard(app, window, dt, allowSceneKeyboard);

    if (Component* selected = FindById(app.components, app.selectedId); selected != nullptr) {
      UpdateSelectedTransformKeyboard(*selected, window, dt, allowSceneKeyboard);
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
      double mx = 0.0;
      double my = 0.0;
      glfwGetCursorPos(window, &mx, &my);

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

      static bool wasLeftDown = false;
      bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      if (leftDown && !wasLeftDown) {
        if (fbW > 0 && fbH > 0) {
          const glm::mat4 view = app.camera.ViewMatrix();
          const glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(fbW) / static_cast<float>(fbH), 0.1f, 500.0f);
          Ray ray = BuildRayFromScreen(mx, my, fbW, fbH, view, proj);
          auto picked = PickComponent(ray, app.components);
          app.selectedId = picked.value_or(0);
        }
      }
      wasLeftDown = leftDown;
    }

    if (allowSceneMouse && app.pendingScroll != 0.0f) {
      app.camera.Zoom(app.pendingScroll);
    }
    app.pendingScroll = 0.0f;

    renderer.DrawScene(app.components, app.selectedId, app.camera, fbW, fbH);

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
