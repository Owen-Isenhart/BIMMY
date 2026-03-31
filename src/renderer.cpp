#include "renderer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace {
struct MaterialTexturePaths {
  const char* albedo = nullptr;
  const char* normal = nullptr;
  const char* roughness = nullptr;
  const char* ao = nullptr;
  const char* metallic = nullptr;
  bool useMetalFallback = false;
};

enum class TextureSlot : std::uint8_t {
  Albedo = 0,
  Normal = 1,
  Metallic = 2,
  Roughness = 3,
  Ao = 4,
};

constexpr std::uint8_t kSlotAlbedoBit = 1u << 0;
constexpr std::uint8_t kSlotNormalBit = 1u << 1;
constexpr std::uint8_t kSlotMetallicBit = 1u << 2;
constexpr std::uint8_t kSlotRoughnessBit = 1u << 3;
constexpr std::uint8_t kSlotAoBit = 1u << 4;

std::uint8_t SlotBit(TextureSlot slot) {
  switch (slot) {
    case TextureSlot::Albedo:
      return kSlotAlbedoBit;
    case TextureSlot::Normal:
      return kSlotNormalBit;
    case TextureSlot::Metallic:
      return kSlotMetallicBit;
    case TextureSlot::Roughness:
      return kSlotRoughnessBit;
    case TextureSlot::Ao:
      return kSlotAoBit;
  }
  return 0;
}

struct TextureDecodeRequest {
  MaterialType type;
  TextureSlot slot;
  std::string path;
};

struct TextureDecodedData {
  MaterialType type;
  TextureSlot slot;
  bool success = false;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<unsigned char> pixels;
};

std::thread gTextureWorker;
std::mutex gDecodeMutex;
std::condition_variable gDecodeCv;
std::deque<TextureDecodeRequest> gDecodeQueue;

std::mutex gUploadMutex;
std::deque<TextureDecodedData> gUploadQueue;

std::unordered_map<MaterialType, std::uint8_t> gRequestedSlots;
std::unordered_map<MaterialType, std::uint8_t> gLoadedSlots;
std::atomic<bool> gWorkerRunning = false;

GLuint gDefaultAlbedo = 0;
GLuint gDefaultNormal = 0;
GLuint gDefaultRoughness = 0;
GLuint gDefaultAo = 0;
GLuint gDefaultMetallicDielectric = 0;
GLuint gDefaultMetallicMetal = 0;

constexpr const char* kDefaultVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoords;

void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorldPos = world.xyz;
  vNormal = normalize(uNormalMat * aNormal);
  vTexCoords = aTexCoords;
  gl_Position = uProj * uView * world;
}
)";

constexpr const char* kDefaultFragmentShader = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoords;

uniform vec3 uCameraPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform float uUvScale;
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform bool uUseUnlitColor;
uniform vec3 uUnlitColor;

out vec4 FragColor;

const float PI = 3.14159265359;

vec3 getNormalFromMap(vec2 uv) {
  vec3 tangentNormal = texture(normalMap, uv).xyz * 2.0 - 1.0;

  vec3 Q1 = dFdx(vWorldPos);
  vec3 Q2 = dFdy(vWorldPos);
  vec2 st1 = dFdx(uv);
  vec2 st2 = dFdy(uv);

  vec3 N = normalize(vNormal);
  vec3 T = normalize(Q1 * st2.y - Q2 * st1.y);
  vec3 B = -normalize(cross(N, T));
  mat3 TBN = mat3(T, B, N);

  return normalize(TBN * tangentNormal);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;

  float num = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;

  return num / max(denom, 0.0000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = (r * r) / 8.0;

  float num = NdotV;
  float denom = NdotV * (1.0 - k) + k;

  return num / max(denom, 0.0000001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = GeometrySchlickGGX(NdotV, roughness);
  float ggx1 = GeometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
  if (uUseUnlitColor) {
    FragColor = vec4(uUnlitColor, 1.0);
    return;
  }

  vec2 uv = vTexCoords * uUvScale;

  vec3 albedo = pow(texture(albedoMap, uv).rgb, vec3(2.2));
  float metallic = clamp(texture(metallicMap, uv).r, 0.0, 1.0);
  float roughness = clamp(texture(roughnessMap, uv).r, 0.05, 1.0);
  float ao = clamp(texture(aoMap, uv).r, 0.0, 1.0);

  vec3 N = getNormalFromMap(uv);
  vec3 L = normalize(lightPos - vWorldPos);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 H = normalize(L + V);

  float distance = length(lightPos - vWorldPos);
  float attenuation = 1.0 / max(distance * distance, 0.0001);
  vec3 radiance = lightColor * attenuation;

  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, metallic);

  float NDF = DistributionGGX(N, H, roughness);
  float G = GeometrySmith(N, V, L, roughness);
  vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

  vec3 numerator = NDF * G * F;
  float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
  vec3 specular = numerator / max(denominator, 0.0001);

  vec3 kS = F;
  vec3 kD = vec3(1.0) - kS;
  kD *= (1.0 - metallic);

  float NdotL = max(dot(N, L), 0.0);
  vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

  vec3 ambient = vec3(0.03) * albedo * ao;
  vec3 color = ambient + Lo;

  color = color / (color + vec3(1.0));
  color = pow(color, vec3(1.0 / 2.2));

  FragColor = vec4(color, 1.0);
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

const MaterialTexturePaths& GetMaterialTexturePaths(MaterialType type) {
  static const MaterialTexturePaths kSheetMetal {
    "box_profile_metal_sheet_4k/textures/box_profile_metal_sheet_diff_4k.png",
    "box_profile_metal_sheet_4k/textures/box_profile_metal_sheet_nor_gl_4k.png",
    "box_profile_metal_sheet_4k/textures/box_profile_metal_sheet_rough_4k.png",
    "box_profile_metal_sheet_4k/textures/box_profile_metal_sheet_ao_4k.png",
    "box_profile_metal_sheet_4k/textures/box_profile_metal_sheet_metal_4k.png",
    true,
  };

  static const MaterialTexturePaths kGrass {
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_diff_4k.png",
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_nor_gl_4k.png",
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_rough_4k.png",
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_ao_4k.png",
    nullptr,
    false,
  };

  static const MaterialTexturePaths kConcrete {
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_diff_4k.png",
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_nor_gl_4k.png",
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_rough_4k.png",
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_ao_4k.png",
    nullptr,
    false,
  };

  static const MaterialTexturePaths kRustedMetal {
    "green_metal_rust_4k/textures/green_metal_rust_diff_4k.png",
    "green_metal_rust_4k/textures/green_metal_rust_nor_gl_4k.png",
    "green_metal_rust_4k/textures/green_metal_rust_rough_4k.png",
    "green_metal_rust_4k/textures/green_metal_rust_ao_4k.png",
    nullptr,
    true,
  };

  static const MaterialTexturePaths kBrick {
    "red_brick_4k/textures/red_brick_diff_4k.png",
    "red_brick_4k/textures/red_brick_nor_gl_4k.png",
    "red_brick_4k/textures/red_brick_rough_4k.png",
    "red_brick_4k/textures/red_brick_ao_4k.png",
    nullptr,
    false,
  };

  static const MaterialTexturePaths kRoof {
    "roof_slates_03_4k/textures/roof_slates_03_diff_4k.png",
    "roof_slates_03_4k/textures/roof_slates_03_nor_gl_4k.png",
    "roof_slates_03_4k/textures/roof_slates_03_rough_4k.png",
    "roof_slates_03_4k/textures/roof_slates_03_ao_4k.png",
    nullptr,
    false,
  };

  static const MaterialTexturePaths kWood {
    "weathered_brown_planks_4k/textures/weathered_brown_planks_diff_4k.png",
    "weathered_brown_planks_4k/textures/weathered_brown_planks_nor_gl_4k.png",
    "weathered_brown_planks_4k/textures/weathered_brown_planks_rough_4k.png",
    "weathered_brown_planks_4k/textures/weathered_brown_planks_ao_4k.png",
    nullptr,
    false,
  };

  switch (type) {
    case MaterialType::SheetMetal:
      return kSheetMetal;
    case MaterialType::Grass:
      return kGrass;
    case MaterialType::Concrete:
      return kConcrete;
    case MaterialType::RustedMetal:
      return kRustedMetal;
    case MaterialType::Brick:
      return kBrick;
    case MaterialType::Roof:
      return kRoof;
    case MaterialType::Wood:
      return kWood;
  }

  return kConcrete;
}

GLuint CreateSolidTexture(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255) {
  const std::uint8_t pixel[4] = {r, g, b, a};

  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
  glGenerateMipmap(GL_TEXTURE_2D);

  return texture;
}

void InitializeFallbackTextures() {
  if (gDefaultAlbedo != 0) {
    return;
  }

  gDefaultAlbedo = CreateSolidTexture(204, 204, 204);
  gDefaultNormal = CreateSolidTexture(128, 128, 255);
  gDefaultRoughness = CreateSolidTexture(180, 180, 180);
  gDefaultAo = CreateSolidTexture(255, 255, 255);
  gDefaultMetallicDielectric = CreateSolidTexture(0, 0, 0);
  gDefaultMetallicMetal = CreateSolidTexture(255, 255, 255);
}

void ReleaseFallbackTextures() {
  const GLuint ids[] = {
    gDefaultAlbedo,
    gDefaultNormal,
    gDefaultRoughness,
    gDefaultAo,
    gDefaultMetallicDielectric,
    gDefaultMetallicMetal,
  };

  for (GLuint id : ids) {
    if (id != 0) {
      glDeleteTextures(1, &id);
    }
  }

  gDefaultAlbedo = 0;
  gDefaultNormal = 0;
  gDefaultRoughness = 0;
  gDefaultAo = 0;
  gDefaultMetallicDielectric = 0;
  gDefaultMetallicMetal = 0;
}

std::filesystem::path ResolveTexturePath(const char* path) {
  std::vector<std::filesystem::path> roots;

#ifdef BIMMY_TEXTURE_DIR
  roots.emplace_back(BIMMY_TEXTURE_DIR);
#endif
  roots.emplace_back("textures");
  roots.emplace_back("../textures");

  for (const auto& root : roots) {
    const auto full = root / path;
    if (std::filesystem::exists(full)) {
      return full;
    }
  }

  return std::filesystem::path(path);
}

GLenum TextureFormatFromChannels(int channels) {
  if (channels == 1) return GL_RED;
  if (channels == 4) return GL_RGBA;
  return GL_RGB;
}

GLenum TextureInternalFormatFromChannels(int channels) {
  if (channels == 1) return GL_R8;
  if (channels == 4) return GL_RGBA8;
  return GL_RGB8;
}

GLuint UploadTextureFromPixels(const TextureDecodedData& decoded) {
  if (!decoded.success || decoded.pixels.empty() || decoded.width <= 0 || decoded.height <= 0) {
    return 0;
  }

  const GLenum format = TextureFormatFromChannels(decoded.channels);
  const GLenum internalFormat = TextureInternalFormatFromChannels(decoded.channels);

  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, decoded.width, decoded.height, 0, format, GL_UNSIGNED_BYTE, decoded.pixels.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  return texture;
}

const char* PathForSlot(const MaterialTexturePaths& paths, TextureSlot slot) {
  switch (slot) {
    case TextureSlot::Albedo:
      return paths.albedo;
    case TextureSlot::Normal:
      return paths.normal;
    case TextureSlot::Metallic:
      return paths.metallic;
    case TextureSlot::Roughness:
      return paths.roughness;
    case TextureSlot::Ao:
      return paths.ao;
  }
  return nullptr;
}

void TextureWorkerMain() {
  while (gWorkerRunning.load()) {
    TextureDecodeRequest request;
    {
      std::unique_lock<std::mutex> lock(gDecodeMutex);
      gDecodeCv.wait(lock, [] { return !gWorkerRunning.load() || !gDecodeQueue.empty(); });
      if (!gWorkerRunning.load() && gDecodeQueue.empty()) {
        return;
      }
      request = std::move(gDecodeQueue.front());
      gDecodeQueue.pop_front();
    }

    TextureDecodedData decoded;
    decoded.type = request.type;
    decoded.slot = request.slot;

    const std::filesystem::path resolved = ResolveTexturePath(request.path.c_str());
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_set_flip_vertically_on_load(1);
    unsigned char* data = stbi_load(resolved.string().c_str(), &width, &height, &channels, 0);
    if (data != nullptr && width > 0 && height > 0) {
      const std::size_t sizeBytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
      decoded.success = true;
      decoded.width = width;
      decoded.height = height;
      decoded.channels = channels;
      decoded.pixels.assign(data, data + sizeBytes);
      stbi_image_free(data);
    } else {
      if (data != nullptr) {
        stbi_image_free(data);
      }
      std::cerr << "Texture decode failed: " << resolved << "\n";
    }

    {
      std::lock_guard<std::mutex> lock(gUploadMutex);
      gUploadQueue.push_back(std::move(decoded));
    }
  }
}

void StartTextureWorker() {
  if (gWorkerRunning.load()) {
    return;
  }
  gWorkerRunning = true;
  gTextureWorker = std::thread(TextureWorkerMain);
}

void StopTextureWorker() {
  if (!gWorkerRunning.load()) {
    return;
  }

  gWorkerRunning = false;
  gDecodeCv.notify_all();
  if (gTextureWorker.joinable()) {
    gTextureWorker.join();
  }

  {
    std::lock_guard<std::mutex> lock(gDecodeMutex);
    gDecodeQueue.clear();
  }
  {
    std::lock_guard<std::mutex> lock(gUploadMutex);
    gUploadQueue.clear();
  }
}

GLuint LoadTexture(const char* path) {
  const std::filesystem::path resolved = ResolveTexturePath(path);

  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_set_flip_vertically_on_load(1);
  unsigned char* data = stbi_load(resolved.string().c_str(), &width, &height, &channels, 0);
  if (data == nullptr) {
    std::cerr << "Texture load failed: " << resolved << "\n";
    return 0;
  }

  GLenum format = GL_RGB;
  GLenum internalFormat = GL_RGB8;
  if (channels == 1) {
    format = GL_RED;
    internalFormat = GL_R8;
  } else if (channels == 3) {
    format = GL_RGB;
    internalFormat = GL_RGB8;
  } else if (channels == 4) {
    format = GL_RGBA;
    internalFormat = GL_RGBA8;
  }

  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);

  stbi_image_free(data);
  return texture;
}

void EnsureMaterialTexturesLoaded(MaterialType type, MaterialDefinition& mat) {
  InitializeFallbackTextures();
  const MaterialTexturePaths& paths = GetMaterialTexturePaths(type);

  if (mat.albedoMap == 0) mat.albedoMap = gDefaultAlbedo;
  if (mat.normalMap == 0) mat.normalMap = gDefaultNormal;
  if (mat.roughnessMap == 0) mat.roughnessMap = gDefaultRoughness;
  if (mat.aoMap == 0) mat.aoMap = gDefaultAo;
  if (mat.metallicMap == 0) mat.metallicMap = paths.useMetalFallback ? gDefaultMetallicMetal : gDefaultMetallicDielectric;

  auto QueueSlot = [&](TextureSlot slot) {
    const std::uint8_t bit = SlotBit(slot);
    const std::uint8_t loaded = gLoadedSlots[type];
    const std::uint8_t requested = gRequestedSlots[type];
    if ((loaded & bit) != 0 || (requested & bit) != 0) {
      return;
    }

    const char* path = PathForSlot(paths, slot);
    if (path == nullptr) {
      gLoadedSlots[type] |= bit;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(gDecodeMutex);
      gDecodeQueue.push_back(TextureDecodeRequest{type, slot, path});
      gRequestedSlots[type] |= bit;
    }
    gDecodeCv.notify_one();
  };

  QueueSlot(TextureSlot::Albedo);
  QueueSlot(TextureSlot::Normal);
  QueueSlot(TextureSlot::Metallic);
  QueueSlot(TextureSlot::Roughness);
  QueueSlot(TextureSlot::Ao);
}

void ApplyUploadedTexture(MaterialType type, TextureSlot slot, GLuint texture) {
  auto& catalog = MutableMaterialCatalog();
  const auto it = catalog.find(type);
  if (it == catalog.end() || texture == 0) {
    return;
  }

  MaterialDefinition& mat = it->second;
  GLuint* target = nullptr;
  GLuint fallback = 0;

  switch (slot) {
    case TextureSlot::Albedo:
      target = &mat.albedoMap;
      fallback = gDefaultAlbedo;
      break;
    case TextureSlot::Normal:
      target = &mat.normalMap;
      fallback = gDefaultNormal;
      break;
    case TextureSlot::Metallic:
      target = &mat.metallicMap;
      fallback = gDefaultMetallicDielectric;
      break;
    case TextureSlot::Roughness:
      target = &mat.roughnessMap;
      fallback = gDefaultRoughness;
      break;
    case TextureSlot::Ao:
      target = &mat.aoMap;
      fallback = gDefaultAo;
      break;
  }

  if (target == nullptr) {
    glDeleteTextures(1, &texture);
    return;
  }

  if (*target != 0 && *target != fallback && *target != gDefaultMetallicMetal && *target != gDefaultMetallicDielectric) {
    glDeleteTextures(1, target);
  }
  *target = texture;
  gLoadedSlots[type] |= SlotBit(slot);
}

void PumpDecodedTextureUploads(int maxUploads) {
  std::deque<TextureDecodedData> local;
  {
    std::lock_guard<std::mutex> lock(gUploadMutex);
    const int count = std::min(maxUploads, static_cast<int>(gUploadQueue.size()));
    for (int i = 0; i < count; ++i) {
      local.push_back(std::move(gUploadQueue.front()));
      gUploadQueue.pop_front();
    }
  }

  for (auto& decoded : local) {
    GLuint uploaded = 0;
    if (decoded.success) {
      uploaded = UploadTextureFromPixels(decoded);
    }

    if (uploaded != 0) {
      ApplyUploadedTexture(decoded.type, decoded.slot, uploaded);
    } else {
      gLoadedSlots[decoded.type] |= SlotBit(decoded.slot);
    }
  }
}

void InitializeMaterialCatalog() {
  auto& catalog = MutableMaterialCatalog();
  if (!catalog.empty()) {
    return;
  }

  catalog[MaterialType::SheetMetal] = {"Sheet Metal", 320.0f};
  catalog[MaterialType::Grass] = {"Grass", 30.0f};
  catalog[MaterialType::Concrete] = {"Concrete", 140.0f};
  catalog[MaterialType::RustedMetal] = {"Rusted Metal", 300.0f};
  catalog[MaterialType::Brick] = {"Brick", 160.0f};
  catalog[MaterialType::Roof] = {"Roof Slate", 180.0f};
  catalog[MaterialType::Wood] = {"Wood", 95.0f};

  gRequestedSlots.clear();
  gLoadedSlots.clear();

  for (const auto& [type, _] : catalog) {
    gRequestedSlots[type] = 0;
    gLoadedSlots[type] = 0;
  }
}

void BindMaterialTextures(GLuint program, const MaterialDefinition& mat) {
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, mat.albedoMap);
  glUniform1i(glGetUniformLocation(program, "albedoMap"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, mat.normalMap);
  glUniform1i(glGetUniformLocation(program, "normalMap"), 1);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, mat.metallicMap);
  glUniform1i(glGetUniformLocation(program, "metallicMap"), 2);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, mat.roughnessMap);
  glUniform1i(glGetUniformLocation(program, "roughnessMap"), 3);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, mat.aoMap);
  glUniform1i(glGetUniformLocation(program, "aoMap"), 4);
}

void ReleaseMaterialTextures() {
  auto& catalog = MutableMaterialCatalog();
  std::unordered_set<GLuint> textures;

  for (const auto& [_, mat] : catalog) {
    if (mat.albedoMap != 0 && mat.albedoMap != gDefaultAlbedo) textures.insert(mat.albedoMap);
    if (mat.normalMap != 0 && mat.normalMap != gDefaultNormal) textures.insert(mat.normalMap);
    if (mat.metallicMap != 0 && mat.metallicMap != gDefaultMetallicDielectric && mat.metallicMap != gDefaultMetallicMetal) textures.insert(mat.metallicMap);
    if (mat.roughnessMap != 0 && mat.roughnessMap != gDefaultRoughness) textures.insert(mat.roughnessMap);
    if (mat.aoMap != 0 && mat.aoMap != gDefaultAo) textures.insert(mat.aoMap);
  }

  for (GLuint t : textures) {
    glDeleteTextures(1, &t);
  }

  catalog.clear();
  gRequestedSlots.clear();
  gLoadedSlots.clear();
  ReleaseFallbackTextures();
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

std::unordered_map<MaterialType, MaterialDefinition>& MutableMaterialCatalog() {
  static std::unordered_map<MaterialType, MaterialDefinition> catalog;
  return catalog;
}

const std::unordered_map<MaterialType, MaterialDefinition>& MaterialCatalog() {
  return MutableMaterialCatalog();
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

  InitializeMaterialCatalog();
  StartTextureWorker();

  for (auto& [type, mat] : MutableMaterialCatalog()) {
    EnsureMaterialTexturesLoaded(type, mat);
  }

  cuboid_ = GpuMesh(CreateCuboidMesh());
  cylinder_ = GpuMesh(CreateCylinderMesh(28));
  prism_ = GpuMesh(CreatePrismMesh());
  plane_ = GpuMesh(CreatePlaneMesh());

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  return true;
}

void Renderer::Shutdown() {
  StopTextureWorker();
  ReleaseMaterialTextures();

  if (program_ != 0) {
    glDeleteProgram(program_);
    program_ = 0;
  }
}

void Renderer::DrawScene(const std::vector<Component>& components, std::uint32_t selectedId, const OrbitCamera& camera, int viewportWidth, int viewportHeight, bool drawFloor) {
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

  PumpDecodedTextureUploads(3);

  glUniformMatrix4fv(glGetUniformLocation(program_, "uView"), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(program_, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
  glUniform3fv(glGetUniformLocation(program_, "uCameraPos"), 1, glm::value_ptr(camPos));
  glUniform3f(glGetUniformLocation(program_, "lightPos"), 7.0f, 10.0f, 8.0f);
  glUniform3f(glGetUniformLocation(program_, "lightColor"), 40.0f, 40.0f, 40.0f);
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

  if (drawFloor) {
    const auto floorIt = MaterialCatalog().find(MaterialType::Grass);
    if (floorIt != MaterialCatalog().end()) {
      EnsureMaterialTexturesLoaded(MaterialType::Grass, MutableMaterialCatalog().at(MaterialType::Grass));
      const glm::mat4 floorModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.001f, 0.0f)) *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(220.0f, 1.0f, 220.0f));
      const glm::mat3 floorNormal = glm::inverseTranspose(glm::mat3(floorModel));

      const MaterialDefinition& floorMat = MaterialCatalog().at(MaterialType::Grass);
      glDisable(GL_CULL_FACE);
      BindMaterialTextures(program_, floorMat);
      glUniform1f(glGetUniformLocation(program_, "uUvScale"), 120.0f);
      glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, glm::value_ptr(floorModel));
      glUniformMatrix3fv(glGetUniformLocation(program_, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(floorNormal));
      plane_.Draw();
      glEnable(GL_CULL_FACE);
    }
  }

  for (const Component& c : components) {
    auto matIt = MutableMaterialCatalog().find(c.material);
    if (matIt == MaterialCatalog().end()) {
      continue;
    }

    EnsureMaterialTexturesLoaded(c.material, matIt->second);

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
    glUniform1f(glGetUniformLocation(program_, "uUvScale"), 1.0f);

    BindMaterialTextures(program_, mat);

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
