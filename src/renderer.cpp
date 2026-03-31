#include "renderer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace {
constexpr const char* kDefaultVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;

out vec3 vWorldPos;
out vec3 vNormal;

void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorldPos = world.xyz;
  vNormal = normalize(uNormalMat * aNormal);
  gl_Position = uProj * uView * world;
}
)";

constexpr const char* kDefaultFragmentShader = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uCameraPos;
uniform vec3 uBaseColor;
uniform float uSpecularStrength;
uniform float uRoughness;
uniform bool uUseUnlitColor;
uniform vec3 uUnlitColor;

out vec4 FragColor;

void main() {
  if (uUseUnlitColor) {
    FragColor = vec4(uUnlitColor, 1.0);
    return;
  }

  vec3 lightPos = vec3(7.0, 10.0, 8.0);
  vec3 lightColor = vec3(1.0);

  vec3 N = normalize(vNormal);
  vec3 L = normalize(lightPos - vWorldPos);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 H = normalize(L + V);

  float diff = max(dot(N, L), 0.0);
  float glossPower = mix(8.0, 128.0, 1.0 - clamp(uRoughness, 0.0, 1.0));
  float spec = pow(max(dot(N, H), 0.0), glossPower) * uSpecularStrength;

  vec3 ambient = 0.18 * uBaseColor;
  vec3 lit = ambient + uBaseColor * diff + lightColor * spec;

  FragColor = vec4(lit, 1.0);
}
)";

bool ReadTextFile(const std::filesystem::path& path, std::string& outText) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::ostringstream buffer;
  buffer << file.rdbuf();
  outText = buffer.str();
  return true;
}

bool TryLoadExternalShaders(std::string& outVertex, std::string& outFragment) {
  std::vector<std::filesystem::path> shaderRoots;

#ifdef BIMMY_SHADER_DIR
  shaderRoots.emplace_back(BIMMY_SHADER_DIR);
#endif

  shaderRoots.emplace_back("shaders");
  shaderRoots.emplace_back("../shaders");

  for (const auto& root : shaderRoots) {
    std::string vertex;
    std::string fragment;

    const std::filesystem::path vertexPath = root / "bimmy.vert";
    const std::filesystem::path fragmentPath = root / "bimmy.frag";

    if (!ReadTextFile(vertexPath, vertex) || !ReadTextFile(fragmentPath, fragment)) {
      continue;
    }

    if (vertex.empty() || fragment.empty()) {
      continue;
    }

    outVertex = std::move(vertex);
    outFragment = std::move(fragment);
    std::cout << "Loaded shaders from: " << root << "\n";
    return true;
  }

  return false;
}

GLuint Compile(GLenum type, const std::string& src) {
  GLuint shader = glCreateShader(type);
  const char* source = src.c_str();
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024] = {};
    glGetShaderInfoLog(shader, 1024, nullptr, log);
    std::cerr << "Shader compile error: " << log << "\n";
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}
}

bool Renderer::BuildProgram() {
  std::string vertexSource = kDefaultVertexShader;
  std::string fragmentSource = kDefaultFragmentShader;
  if (!TryLoadExternalShaders(vertexSource, fragmentSource)) {
    std::cerr << "Warning: external shaders not found. Falling back to built-in shaders.\n";
  }

  GLuint vs = Compile(GL_VERTEX_SHADER, vertexSource);
  GLuint fs = Compile(GL_FRAGMENT_SHADER, fragmentSource);
  if (vs == 0 || fs == 0) {
    return false;
  }

  program_ = glCreateProgram();
  glAttachShader(program_, vs);
  glAttachShader(program_, fs);
  glLinkProgram(program_);

  glDeleteShader(vs);
  glDeleteShader(fs);

  GLint ok = 0;
  glGetProgramiv(program_, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024] = {};
    glGetProgramInfoLog(program_, 1024, nullptr, log);
    std::cerr << "Program link error: " << log << "\n";
    return false;
  }

  return true;
}

bool Renderer::Initialize() {
  if (!BuildProgram()) {
    return false;
  }

  cuboid_ = GpuMesh(CreateCuboidMesh());
  cylinder_ = GpuMesh(CreateCylinderMesh(28));
  prism_ = GpuMesh(CreatePrismMesh());

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  return true;
}

void Renderer::Shutdown() {
  if (program_ != 0) {
    glDeleteProgram(program_);
    program_ = 0;
  }
}

void Renderer::DrawScene(const std::vector<Component>& components, std::uint32_t selectedId, const OrbitCamera& camera, int viewportWidth, int viewportHeight) {
  if (viewportWidth <= 0 || viewportHeight <= 0) {
    return;
  }

  glViewport(0, 0, viewportWidth, viewportHeight);
  glClearColor(0.08f, 0.10f, 0.12f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const glm::mat4 view = camera.ViewMatrix();
  const glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight), 0.1f, 500.0f);
  const glm::vec3 camPos = camera.Position();

  glUseProgram(program_);

  glUniformMatrix4fv(glGetUniformLocation(program_, "uView"), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(program_, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
  glUniform3fv(glGetUniformLocation(program_, "uCameraPos"), 1, glm::value_ptr(camPos));
  glUniform1i(glGetUniformLocation(program_, "uUseUnlitColor"), 0);

  auto DrawMeshForType = [&](GeometryType type) {
    if (type == GeometryType::Cuboid) {
      cuboid_.Draw();
    } else if (type == GeometryType::Cylinder) {
      cylinder_.Draw();
    } else {
      prism_.Draw();
    }
  };

  for (const Component& c : components) {
    const auto matIt = MaterialCatalog().find(c.material);
    if (matIt == MaterialCatalog().end()) {
      continue;
    }

    const MaterialDefinition& mat = matIt->second;
    glm::mat4 model = c.transform.ModelMatrix();

    if (c.geometry == GeometryType::Cuboid) {
      model = glm::scale(model, c.dimensions);
    } else if (c.geometry == GeometryType::Cylinder) {
      model = glm::scale(model, glm::vec3(c.dimensions.x * 2.0f, c.dimensions.y, c.dimensions.x * 2.0f));
    } else if (c.geometry == GeometryType::Prism) {
      model = glm::scale(model, c.dimensions);
    }

    const glm::mat3 normalMat = glm::inverseTranspose(glm::mat3(model));

    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix3fv(glGetUniformLocation(program_, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(normalMat));
    glUniform3fv(glGetUniformLocation(program_, "uBaseColor"), 1, glm::value_ptr(mat.albedo));
    glUniform1f(glGetUniformLocation(program_, "uSpecularStrength"), mat.specularStrength);
    glUniform1f(glGetUniformLocation(program_, "uRoughness"), mat.roughness);

    DrawMeshForType(c.geometry);

    if (c.id == selectedId) {
      glEnable(GL_POLYGON_OFFSET_LINE);
      glPolygonOffset(-1.0f, -1.0f);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glDisable(GL_CULL_FACE);
      glLineWidth(2.0f);

      glUniform1i(glGetUniformLocation(program_, "uUseUnlitColor"), 1);
      glUniform3f(glGetUniformLocation(program_, "uUnlitColor"), 1.0f, 0.85f, 0.2f);
      DrawMeshForType(c.geometry);

      glUniform1i(glGetUniformLocation(program_, "uUseUnlitColor"), 0);
      glLineWidth(1.0f);
      glEnable(GL_CULL_FACE);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glDisable(GL_POLYGON_OFFSET_LINE);
    }
  }
}
