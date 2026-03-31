#include "mesh.hpp"

#include <cmath>
#include <utility>

namespace {
void AppendQuad(MeshCpu& mesh, const glm::vec3& n, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d) {
  const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
  mesh.vertices.push_back({a, n, glm::vec2(0.0f, 0.0f)});
  mesh.vertices.push_back({b, n, glm::vec2(1.0f, 0.0f)});
  mesh.vertices.push_back({c, n, glm::vec2(1.0f, 1.0f)});
  mesh.vertices.push_back({d, n, glm::vec2(0.0f, 1.0f)});
  mesh.indices.insert(mesh.indices.end(), {base + 0, base + 1, base + 2, base + 0, base + 2, base + 3});
}
}

GpuMesh::GpuMesh(const MeshCpu& mesh) {
  indexCount_ = static_cast<GLsizei>(mesh.indices.size());

  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  glGenBuffers(1, &ebo_);

  glBindVertexArray(vao_);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(Vertex)), mesh.vertices.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(std::uint32_t)), mesh.indices.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));

  glBindVertexArray(0);
}

GpuMesh::~GpuMesh() {
  if (ebo_ != 0) glDeleteBuffers(1, &ebo_);
  if (vbo_ != 0) glDeleteBuffers(1, &vbo_);
  if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
}

GpuMesh::GpuMesh(GpuMesh&& other) noexcept {
  *this = std::move(other);
}

GpuMesh& GpuMesh::operator=(GpuMesh&& other) noexcept {
  if (this == &other) return *this;
  std::swap(vao_, other.vao_);
  std::swap(vbo_, other.vbo_);
  std::swap(ebo_, other.ebo_);
  std::swap(indexCount_, other.indexCount_);
  return *this;
}

void GpuMesh::Draw() const {
  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
}

MeshCpu CreateCuboidMesh() {
  MeshCpu mesh;
  const glm::vec3 p000(-0.5f, -0.5f, -0.5f);
  const glm::vec3 p001(-0.5f, -0.5f, 0.5f);
  const glm::vec3 p010(-0.5f, 0.5f, -0.5f);
  const glm::vec3 p011(-0.5f, 0.5f, 0.5f);
  const glm::vec3 p100(0.5f, -0.5f, -0.5f);
  const glm::vec3 p101(0.5f, -0.5f, 0.5f);
  const glm::vec3 p110(0.5f, 0.5f, -0.5f);
  const glm::vec3 p111(0.5f, 0.5f, 0.5f);

  AppendQuad(mesh, {0, 0, 1}, p001, p101, p111, p011);
  AppendQuad(mesh, {0, 0, -1}, p100, p000, p010, p110);
  AppendQuad(mesh, {1, 0, 0}, p101, p100, p110, p111);
  AppendQuad(mesh, {-1, 0, 0}, p000, p001, p011, p010);
  AppendQuad(mesh, {0, 1, 0}, p011, p111, p110, p010);
  AppendQuad(mesh, {0, -1, 0}, p000, p100, p101, p001);

  return mesh;
}

MeshCpu CreateCylinderMesh(int segments) {
  MeshCpu mesh;
  const float halfH = 0.5f;
  const float kPi = 3.14159265359f;
  const float step = 2.0f * kPi / static_cast<float>(segments);

  for (int i = 0; i < segments; ++i) {
    const float a0 = i * step;
    const float a1 = (i + 1) * step;
    const float u0 = static_cast<float>(i) / static_cast<float>(segments);
    const float u1 = static_cast<float>(i + 1) / static_cast<float>(segments);
    glm::vec3 p0(cosf(a0) * 0.5f, -halfH, sinf(a0) * 0.5f);
    glm::vec3 p1(cosf(a1) * 0.5f, -halfH, sinf(a1) * 0.5f);
    glm::vec3 p2(cosf(a1) * 0.5f, halfH, sinf(a1) * 0.5f);
    glm::vec3 p3(cosf(a0) * 0.5f, halfH, sinf(a0) * 0.5f);
    glm::vec3 n0 = glm::normalize(glm::vec3(p0.x, 0.0f, p0.z));
    glm::vec3 n1 = glm::normalize(glm::vec3(p1.x, 0.0f, p1.z));

    std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({p0, n0, glm::vec2(u0, 0.0f)});
    mesh.vertices.push_back({p1, n1, glm::vec2(u1, 0.0f)});
    mesh.vertices.push_back({p2, n1, glm::vec2(u1, 1.0f)});
    mesh.vertices.push_back({p3, n0, glm::vec2(u0, 1.0f)});
    mesh.indices.insert(mesh.indices.end(), {base + 0, base + 1, base + 2, base + 0, base + 2, base + 3});

    std::uint32_t baseTop = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({glm::vec3(0.0f, halfH, 0.0f), glm::vec3(0, 1, 0), glm::vec2(0.5f, 0.5f)});
    mesh.vertices.push_back({p3, glm::vec3(0, 1, 0), glm::vec2(p3.x + 0.5f, p3.z + 0.5f)});
    mesh.vertices.push_back({p2, glm::vec3(0, 1, 0), glm::vec2(p2.x + 0.5f, p2.z + 0.5f)});
    mesh.indices.insert(mesh.indices.end(), {baseTop + 0, baseTop + 1, baseTop + 2});

    std::uint32_t baseBottom = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({glm::vec3(0.0f, -halfH, 0.0f), glm::vec3(0, -1, 0), glm::vec2(0.5f, 0.5f)});
    mesh.vertices.push_back({p1, glm::vec3(0, -1, 0), glm::vec2(p1.x + 0.5f, p1.z + 0.5f)});
    mesh.vertices.push_back({p0, glm::vec3(0, -1, 0), glm::vec2(p0.x + 0.5f, p0.z + 0.5f)});
    mesh.indices.insert(mesh.indices.end(), {baseBottom + 0, baseBottom + 1, baseBottom + 2});
  }

  return mesh;
}

MeshCpu CreatePrismMesh() {
  MeshCpu mesh;
  const glm::vec3 a(-0.5f, -0.5f, -0.5f);
  const glm::vec3 b(0.5f, -0.5f, -0.5f);
  const glm::vec3 c(0.0f, 0.5f, -0.5f);
  const glm::vec3 a2(-0.5f, -0.5f, 0.5f);
  const glm::vec3 b2(0.5f, -0.5f, 0.5f);
  const glm::vec3 c2(0.0f, 0.5f, 0.5f);

  auto addTri = [&](const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
    glm::vec3 n = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({p0, n, glm::vec2(0.0f, 0.0f)});
    mesh.vertices.push_back({p1, n, glm::vec2(1.0f, 0.0f)});
    mesh.vertices.push_back({p2, n, glm::vec2(0.5f, 1.0f)});
    mesh.indices.insert(mesh.indices.end(), {base + 0, base + 1, base + 2});
  };

  auto addQuad = [&](const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3) {
    glm::vec3 n = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({p0, n, glm::vec2(0.0f, 0.0f)});
    mesh.vertices.push_back({p1, n, glm::vec2(1.0f, 0.0f)});
    mesh.vertices.push_back({p2, n, glm::vec2(1.0f, 1.0f)});
    mesh.vertices.push_back({p3, n, glm::vec2(0.0f, 1.0f)});
    mesh.indices.insert(mesh.indices.end(), {base + 0, base + 1, base + 2, base + 0, base + 2, base + 3});
  };

  addTri(a, b, c);
  addTri(c2, b2, a2);
  addQuad(a, a2, b2, b);
  addQuad(a2, a, c, c2);
  addQuad(b, b2, c2, c);

  return mesh;
}

MeshCpu CreatePlaneMesh() {
  MeshCpu mesh;
  const glm::vec3 n(0.0f, 1.0f, 0.0f);
  const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());

  mesh.vertices.push_back({glm::vec3(-0.5f, 0.0f, -0.5f), n, glm::vec2(0.0f, 0.0f)});
  mesh.vertices.push_back({glm::vec3(0.5f, 0.0f, -0.5f), n, glm::vec2(1.0f, 0.0f)});
  mesh.vertices.push_back({glm::vec3(0.5f, 0.0f, 0.5f), n, glm::vec2(1.0f, 1.0f)});
  mesh.vertices.push_back({glm::vec3(-0.5f, 0.0f, 0.5f), n, glm::vec2(0.0f, 1.0f)});

  mesh.indices.insert(mesh.indices.end(), {base + 0, base + 1, base + 2, base + 0, base + 2, base + 3});
  return mesh;
}
