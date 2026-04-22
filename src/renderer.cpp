#include "renderer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
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
  const char* height = nullptr;
  bool useMetalFallback = false;
};

enum class TextureSlot : std::uint8_t {
  Albedo = 0,
  Normal = 1,
  Metallic = 2,
  Roughness = 3,
  Ao = 4,
  Height = 5,
};

constexpr std::uint8_t kSlotAlbedoBit = 1u << 0;
constexpr std::uint8_t kSlotNormalBit = 1u << 1;
constexpr std::uint8_t kSlotMetallicBit = 1u << 2;
constexpr std::uint8_t kSlotRoughnessBit = 1u << 3;
constexpr std::uint8_t kSlotAoBit = 1u << 4;
constexpr std::uint8_t kSlotHeightBit = 1u << 5;
constexpr std::uint8_t kAllTextureSlotBits = kSlotAlbedoBit | kSlotNormalBit | kSlotMetallicBit | kSlotRoughnessBit | kSlotAoBit | kSlotHeightBit;

std::uint8_t SlotBit(TextureSlot slot) {
  switch (slot) {
    case TextureSlot::Albedo: return kSlotAlbedoBit;
    case TextureSlot::Normal: return kSlotNormalBit;
    case TextureSlot::Metallic: return kSlotMetallicBit;
    case TextureSlot::Roughness: return kSlotRoughnessBit;
    case TextureSlot::Ao: return kSlotAoBit;
    case TextureSlot::Height: return kSlotHeightBit;
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
GLuint gDefaultHeight = 0;

constexpr const char* kDefaultPbrVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;
uniform mat4 uLightViewProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoords;
out vec4 vLightSpacePos;

void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorldPos = world.xyz;
  vNormal = normalize(uNormalMat * aNormal);
  vTexCoords = aTexCoords;
  vLightSpacePos = uLightViewProj * world;
  gl_Position = uProj * uView * world;
}
)";

constexpr const char* kDefaultPbrFragmentShader = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoords;
in vec4 vLightSpacePos;

uniform vec3 uCameraPos;
uniform vec3 uPointLightPos;
uniform vec3 uPointLightColor;
uniform float uPointLightIntensity;
uniform vec3 uDirLightDir;
uniform vec3 uDirLightColor;
uniform float uDirLightIntensity;
uniform vec3 uSkyAmbientColor;
uniform vec3 uGroundAmbientColor;
uniform float uAmbientIntensity;
uniform float uUvScale;
uniform float uAoStrength;
uniform float uParallaxHeightScale;
uniform float uGlobalRoughnessMul;
uniform float uGlobalMetallicMul;
uniform bool uEnableShadows;
uniform int uShadowPcfRadius;
uniform bool uUseUnlitColor;
uniform vec3 uUnlitColor;

uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D heightMap;
uniform sampler2D shadowMap;
uniform samplerCube envMap;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 BrightColor;

const float PI = 3.14159265359;

vec3 GetNormalFromMap(vec2 uv) {
  vec3 tangentNormal = texture(normalMap, uv).xyz * 2.0 - 1.0;

  vec3 q1 = dFdx(vWorldPos);
  vec3 q2 = dFdy(vWorldPos);
  vec2 st1 = dFdx(uv);
  vec2 st2 = dFdy(uv);

  vec3 n = normalize(vNormal);
  vec3 t = normalize(q1 * st2.y - q2 * st1.y);
  vec3 b = -normalize(cross(n, t));
  mat3 tbn = mat3(t, b, n);

  return normalize(tbn * tangentNormal);
}

vec2 ParallaxUv(vec2 uv, vec3 viewDir) {
  float h = texture(heightMap, uv).r;
  float scale = uParallaxHeightScale;
  vec2 p = viewDir.xy / max(viewDir.z, 0.08) * ((h - 0.5) * scale);
  return uv - p;
}

float DistributionGGX(vec3 n, vec3 h, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float nDotH = max(dot(n, h), 0.0);
  float nDotH2 = nDotH * nDotH;

  float num = a2;
  float denom = (nDotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;

  return num / max(denom, 0.0000001);
}

float GeometrySchlickGGX(float nDotV, float roughness) {
  float r = (roughness + 1.0);
  float k = (r * r) / 8.0;

  float num = nDotV;
  float denom = nDotV * (1.0 - k) + k;
  return num / max(denom, 0.0000001);
}

float GeometrySmith(vec3 n, vec3 v, vec3 l, float roughness) {
  float nDotV = max(dot(n, v), 0.0);
  float nDotL = max(dot(n, l), 0.0);
  float ggx2 = GeometrySchlickGGX(nDotV, roughness);
  float ggx1 = GeometrySchlickGGX(nDotL, roughness);
  return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 f0) {
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

float ShadowFactor(vec3 n, vec3 lDir) {
  if (!uEnableShadows) {
    return 1.0;
  }

  vec3 projCoords = vLightSpacePos.xyz / max(vLightSpacePos.w, 0.00001);
  projCoords = projCoords * 0.5 + 0.5;
  if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) {
    return 1.0;
  }

  float currentDepth = projCoords.z;
  float bias = max(0.001 * (1.0 - dot(n, -lDir)), 0.0003);
  vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));

  float sum = 0.0;
  int samples = 0;
  for (int x = -uShadowPcfRadius; x <= uShadowPcfRadius; ++x) {
    for (int y = -uShadowPcfRadius; y <= uShadowPcfRadius; ++y) {
      float pcfDepth = texture(shadowMap, projCoords.xy + vec2(float(x), float(y)) * texelSize).r;
      sum += (currentDepth - bias) <= pcfDepth ? 1.0 : 0.0;
      samples += 1;
    }
  }

  return sum / max(float(samples), 1.0);
}

vec3 EvalPbr(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo, float metallic, float roughness) {
  vec3 h = normalize(v + l);
  vec3 f0 = mix(vec3(0.04), albedo, metallic);

  float ndf = DistributionGGX(n, h, roughness);
  float g = GeometrySmith(n, v, l, roughness);
  vec3 f = FresnelSchlick(max(dot(h, v), 0.0), f0);

  vec3 numerator = ndf * g * f;
  float denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0);
  vec3 specular = numerator / max(denominator, 0.0001);

  vec3 kS = f;
  vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
  float nDotL = max(dot(n, l), 0.0);
  return (kD * albedo / PI + specular) * radiance * nDotL;
}

void main() {
  if (uUseUnlitColor) {
    FragColor = vec4(uUnlitColor, 1.0);
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 nGeom = normalize(vNormal);
  vec3 vDir = normalize(uCameraPos - vWorldPos);

  vec2 uv = vTexCoords * uUvScale;
  vec2 uvP = ParallaxUv(uv, vDir);

  vec3 albedo = texture(albedoMap, uvP).rgb;
  float metallic = clamp(texture(metallicMap, uvP).r * uGlobalMetallicMul, 0.0, 1.0);
  float roughness = clamp(texture(roughnessMap, uvP).r * uGlobalRoughnessMul, 0.05, 1.0);
  float ao = clamp(texture(aoMap, uvP).r, 0.0, 1.0);
  ao = mix(1.0, ao, uAoStrength);

  vec3 n = GetNormalFromMap(uvP);

  vec3 lPoint = normalize(uPointLightPos - vWorldPos);
  float distance = length(uPointLightPos - vWorldPos);
  float attenuation = 1.0 / max(distance * distance, 0.0001);
  vec3 radiancePoint = uPointLightColor * (uPointLightIntensity * attenuation);

  vec3 lDir = normalize(-uDirLightDir);
  vec3 radianceDir = uDirLightColor * uDirLightIntensity;
  float shadow = ShadowFactor(n, lDir);

  vec3 lo = EvalPbr(n, vDir, lPoint, radiancePoint, albedo, metallic, roughness);
  lo += EvalPbr(n, vDir, lDir, radianceDir * shadow, albedo, metallic, roughness);

  float hemi = n.y * 0.5 + 0.5;
  vec3 hemiAmbient = mix(uGroundAmbientColor, uSkyAmbientColor, hemi) * uAmbientIntensity;

  vec3 reflectDir = reflect(-vDir, n);
  vec3 envSpec = texture(envMap, reflectDir).rgb;
  vec3 envDiff = texture(envMap, nGeom).rgb;
  vec3 f0 = mix(vec3(0.04), albedo, metallic);
  vec3 f = FresnelSchlick(max(dot(n, vDir), 0.0), f0);
  vec3 ibl = envDiff * albedo * (1.0 - metallic) + envSpec * f * (1.0 - roughness * 0.6);

  vec3 color = lo + (hemiAmbient * albedo * ao) + ibl * 0.12;

  FragColor = vec4(color, 1.0);

  float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
  if (luminance > 1.0) {
    BrightColor = vec4(color, 1.0);
  } else {
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
  }
}
)";

constexpr const char* kDefaultDepthVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uLightViewProj;

void main() {
  gl_Position = uLightViewProj * uModel * vec4(aPos, 1.0);
}
)";

constexpr const char* kDefaultDepthFragmentShader = R"(
#version 330 core
void main() {
}
)";

constexpr const char* kDefaultQuadVertexShader = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
void main() {
  vUv = aUv;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

constexpr const char* kDefaultBlurFragmentShader = R"(
#version 330 core
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uImage;
uniform bool uHorizontal;

void main() {
  float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
  vec2 texel = 1.0 / vec2(textureSize(uImage, 0));

  vec3 result = texture(uImage, vUv).rgb * weight[0];
  for (int i = 1; i < 5; ++i) {
    vec2 offset = uHorizontal ? vec2(texel.x * i, 0.0) : vec2(0.0, texel.y * i);
    result += texture(uImage, vUv + offset).rgb * weight[i];
    result += texture(uImage, vUv - offset).rgb * weight[i];
  }

  FragColor = vec4(result, 1.0);
}
)";

constexpr const char* kDefaultPostFragmentShader = R"(
#version 330 core
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uExposure;
uniform float uBloomStrength;
uniform bool uEnableBloom;
uniform bool uEnableFxaa;
uniform vec2 uInvViewport;

vec3 ToneMap(vec3 c) {
  vec3 mapped = vec3(1.0) - exp(-c * uExposure);
  return pow(mapped, vec3(1.0 / 2.2));
}

vec3 ApplyFxaa(sampler2D tex, vec2 uv) {
  vec3 rgbNW = texture(tex, uv + vec2(-1.0, -1.0) * uInvViewport).rgb;
  vec3 rgbNE = texture(tex, uv + vec2( 1.0, -1.0) * uInvViewport).rgb;
  vec3 rgbSW = texture(tex, uv + vec2(-1.0,  1.0) * uInvViewport).rgb;
  vec3 rgbSE = texture(tex, uv + vec2( 1.0,  1.0) * uInvViewport).rgb;
  vec3 rgbM  = texture(tex, uv).rgb;

  vec3 luma = vec3(0.299, 0.587, 0.114);
  float lumaNW = dot(rgbNW, luma);
  float lumaNE = dot(rgbNE, luma);
  float lumaSW = dot(rgbSW, luma);
  float lumaSE = dot(rgbSE, luma);
  float lumaM = dot(rgbM, luma);

  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

  vec2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

  float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.0312, 0.0078125);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * uInvViewport;

  vec3 rgbA = 0.5 * (
    texture(tex, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
    texture(tex, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture(tex, uv + dir * -0.5).rgb +
    texture(tex, uv + dir * 0.5).rgb);

  float lumaB = dot(rgbB, luma);
  if (lumaB < lumaMin || lumaB > lumaMax) {
    return rgbA;
  }
  return rgbB;
}

void main() {
  vec3 scene = uEnableFxaa ? ApplyFxaa(uScene, vUv) : texture(uScene, vUv).rgb;
  vec3 bloom = texture(uBloom, vUv).rgb;
  vec3 color = scene;
  if (uEnableBloom) {
    color += bloom * uBloomStrength;
  }
  FragColor = vec4(ToneMap(color), 1.0);
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

bool TryLoadExternalPbrShaders(std::string& outVertex, std::string& outFragment) {
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
    "box_profile_metal_sheet_4k/textures/box_profile_metal_sheet_disp_4k.png",
    true,
  };

  static const MaterialTexturePaths kGrass {
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_diff_4k.png",
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_nor_gl_4k.png",
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_rough_4k.png",
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_ao_4k.png",
    nullptr,
    "brown_mud_leaves_01_4k/textures/brown_mud_leaves_01_disp_4k.png",
    false,
  };

  static const MaterialTexturePaths kConcrete {
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_diff_4k.png",
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_nor_gl_4k.png",
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_rough_4k.png",
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_ao_4k.png",
    nullptr,
    "concrete_floor_worn_001_4k/textures/concrete_floor_worn_001_disp_4k.png",
    false,
  };

  static const MaterialTexturePaths kRustedMetal {
    "green_metal_rust_4k/textures/green_metal_rust_diff_4k.png",
    "green_metal_rust_4k/textures/green_metal_rust_nor_gl_4k.png",
    "green_metal_rust_4k/textures/green_metal_rust_rough_4k.png",
    "green_metal_rust_4k/textures/green_metal_rust_ao_4k.png",
    nullptr,
    "green_metal_rust_4k/textures/green_metal_rust_disp_4k.png",
    true,
  };

  static const MaterialTexturePaths kBrick {
    "red_brick_4k/textures/red_brick_diff_4k.png",
    "red_brick_4k/textures/red_brick_nor_gl_4k.png",
    "red_brick_4k/textures/red_brick_rough_4k.png",
    "red_brick_4k/textures/red_brick_ao_4k.png",
    nullptr,
    "red_brick_4k/textures/red_brick_disp_4k.png",
    false,
  };

  static const MaterialTexturePaths kRoof {
    "roof_slates_03_4k/textures/roof_slates_03_diff_4k.png",
    "roof_slates_03_4k/textures/roof_slates_03_nor_gl_4k.png",
    "roof_slates_03_4k/textures/roof_slates_03_rough_4k.png",
    "roof_slates_03_4k/textures/roof_slates_03_ao_4k.png",
    nullptr,
    "roof_slates_03_4k/textures/roof_slates_03_disp_4k.png",
    false,
  };

  static const MaterialTexturePaths kWood {
    "weathered_brown_planks_4k/textures/weathered_brown_planks_diff_4k.png",
    "weathered_brown_planks_4k/textures/weathered_brown_planks_nor_gl_4k.png",
    "weathered_brown_planks_4k/textures/weathered_brown_planks_rough_4k.png",
    "weathered_brown_planks_4k/textures/weathered_brown_planks_ao_4k.png",
    nullptr,
    "weathered_brown_planks_4k/textures/weathered_brown_planks_disp_4k.png",
    false,
  };

  switch (type) {
    case MaterialType::SheetMetal: return kSheetMetal;
    case MaterialType::Grass: return kGrass;
    case MaterialType::Concrete: return kConcrete;
    case MaterialType::RustedMetal: return kRustedMetal;
    case MaterialType::Brick: return kBrick;
    case MaterialType::Roof: return kRoof;
    case MaterialType::Wood: return kWood;
  }

  return kConcrete;
}

GLuint CreateSolidTexture(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255, bool srgb = false) {
  const std::uint8_t pixel[4] = {r, g, b, a};

  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  const GLenum internal = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
  glTexImage2D(GL_TEXTURE_2D, 0, internal, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
  glGenerateMipmap(GL_TEXTURE_2D);

  return texture;
}

void InitializeFallbackTextures() {
  if (gDefaultAlbedo != 0) {
    return;
  }

  gDefaultAlbedo = CreateSolidTexture(204, 204, 204, 255, true);
  gDefaultNormal = CreateSolidTexture(128, 128, 255);
  gDefaultRoughness = CreateSolidTexture(180, 180, 180);
  gDefaultAo = CreateSolidTexture(255, 255, 255);
  gDefaultMetallicDielectric = CreateSolidTexture(0, 0, 0);
  gDefaultMetallicMetal = CreateSolidTexture(255, 255, 255);
  gDefaultHeight = CreateSolidTexture(128, 128, 128);
}

void ReleaseFallbackTextures() {
  const GLuint ids[] = {
    gDefaultAlbedo,
    gDefaultNormal,
    gDefaultRoughness,
    gDefaultAo,
    gDefaultMetallicDielectric,
    gDefaultMetallicMetal,
    gDefaultHeight,
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
  gDefaultHeight = 0;
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

GLenum TextureInternalFormatFromSlot(TextureSlot slot, int channels) {
  if (slot == TextureSlot::Albedo) {
    if (channels == 4) return GL_SRGB8_ALPHA8;
    return GL_SRGB8;
  }
  if (channels == 1) return GL_R8;
  if (channels == 4) return GL_RGBA8;
  return GL_RGB8;
}

GLuint UploadTextureFromPixels(const TextureDecodedData& decoded) {
  if (!decoded.success || decoded.pixels.empty() || decoded.width <= 0 || decoded.height <= 0) {
    return 0;
  }

  const GLenum format = TextureFormatFromChannels(decoded.channels);
  const GLenum internalFormat = TextureInternalFormatFromSlot(decoded.slot, decoded.channels);

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
    case TextureSlot::Albedo: return paths.albedo;
    case TextureSlot::Normal: return paths.normal;
    case TextureSlot::Metallic: return paths.metallic;
    case TextureSlot::Roughness: return paths.roughness;
    case TextureSlot::Ao: return paths.ao;
    case TextureSlot::Height: return paths.height;
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

void EnsureMaterialTexturesLoaded(MaterialType type, MaterialDefinition& mat) {
  InitializeFallbackTextures();
  const MaterialTexturePaths& paths = GetMaterialTexturePaths(type);

  if (mat.albedoMap == 0) mat.albedoMap = gDefaultAlbedo;
  if (mat.normalMap == 0) mat.normalMap = gDefaultNormal;
  if (mat.roughnessMap == 0) mat.roughnessMap = gDefaultRoughness;
  if (mat.aoMap == 0) mat.aoMap = gDefaultAo;
  if (mat.heightMap == 0) mat.heightMap = gDefaultHeight;
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
  QueueSlot(TextureSlot::Height);
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
    case TextureSlot::Height:
      target = &mat.heightMap;
      fallback = gDefaultHeight;
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

bool AreAllMaterialTexturesReady() {
  const auto& catalog = MaterialCatalog();
  for (const auto& [type, _] : catalog) {
    const auto loadedIt = gLoadedSlots.find(type);
    if (loadedIt == gLoadedSlots.end()) {
      return false;
    }
    if ((loadedIt->second & kAllTextureSlotBits) != kAllTextureSlotBits) {
      return false;
    }
  }
  return !catalog.empty();
}

void WaitForAllTextureUploadsAtStartup() {
  while (!AreAllMaterialTexturesReady()) {
    PumpDecodedTextureUploads(32);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  PumpDecodedTextureUploads(256);
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

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, mat.heightMap);
  glUniform1i(glGetUniformLocation(program, "heightMap"), 5);
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
    if (mat.heightMap != 0 && mat.heightMap != gDefaultHeight) textures.insert(mat.heightMap);
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

GLuint BuildProgramFromSources(const std::string& vertexSource, const std::string& fragmentSource) {
  GLuint vs = Compile(GL_VERTEX_SHADER, vertexSource);
  GLuint fs = Compile(GL_FRAGMENT_SHADER, fragmentSource);
  if (vs == 0 || fs == 0) {
    if (vs != 0) glDeleteShader(vs);
    if (fs != 0) glDeleteShader(fs);
    return 0;
  }

  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  glDeleteShader(vs);
  glDeleteShader(fs);

  GLint ok = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024] = {};
    glGetProgramInfoLog(program, 1024, nullptr, log);
    std::cerr << "Program link error: " << log << "\n";
    glDeleteProgram(program);
    return 0;
  }

  return program;
}

void DeleteIfValid(GLuint& id, bool isProgram) {
  if (id == 0) {
    return;
  }
  if (isProgram) {
    glDeleteProgram(id);
  } else {
    glDeleteTextures(1, &id);
  }
  id = 0;
}

void CreateSimpleEnvironmentCubemap(GLuint& outTex) {
  if (outTex != 0) {
    return;
  }

  glGenTextures(1, &outTex);
  glBindTexture(GL_TEXTURE_CUBE_MAP, outTex);

  const std::array<std::array<std::uint8_t, 3>, 6> faces = {{
    {148, 176, 208},
    {148, 176, 208},
    {168, 194, 224},
    {120, 102, 84},
    {138, 162, 198},
    {138, 162, 198},
  }};

  for (int i = 0; i < 6; ++i) {
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_SRGB8, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, faces[i].data());
  }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

glm::mat4 ComputeDirectionalLightMatrix(const Renderer::Settings& settings) {
  const glm::vec3 lightDir = glm::normalize(settings.directionalDirection);
  const glm::vec3 center(0.0f, 0.0f, 0.0f);
  const glm::vec3 lightPos = center - lightDir * 45.0f;
  const glm::mat4 lightView = glm::lookAt(lightPos, center, glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::mat4 lightProj = glm::ortho(-40.0f, 40.0f, -40.0f, 40.0f, 1.0f, 120.0f);
  return lightProj * lightView;
}
}  // namespace

std::unordered_map<MaterialType, MaterialDefinition>& MutableMaterialCatalog() {
  static std::unordered_map<MaterialType, MaterialDefinition> catalog;
  return catalog;
}

const std::unordered_map<MaterialType, MaterialDefinition>& MaterialCatalog() {
  return MutableMaterialCatalog();
}

bool Renderer::BuildPrograms() {
  std::string vertexSource = kDefaultPbrVertexShader;
  std::string fragmentSource = kDefaultPbrFragmentShader;
  if (!TryLoadExternalPbrShaders(vertexSource, fragmentSource)) {
    std::cerr << "Warning: external shaders not found. Falling back to built-in shaders.\n";
  }

  pbrProgram_ = BuildProgramFromSources(vertexSource, fragmentSource);
  if (pbrProgram_ == 0) {
    return false;
  }

  depthProgram_ = BuildProgramFromSources(kDefaultDepthVertexShader, kDefaultDepthFragmentShader);
  blurProgram_ = BuildProgramFromSources(kDefaultQuadVertexShader, kDefaultBlurFragmentShader);
  postProgram_ = BuildProgramFromSources(kDefaultQuadVertexShader, kDefaultPostFragmentShader);
  return depthProgram_ != 0 && blurProgram_ != 0 && postProgram_ != 0;
}

bool Renderer::InitializeFullscreenQuad() {
  if (quadVao_ != 0) {
    return true;
  }

  const float verts[] = {
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
     1.0f,  1.0f, 1.0f, 1.0f,
  };

  glGenVertexArrays(1, &quadVao_);
  glGenBuffers(1, &quadVbo_);
  glBindVertexArray(quadVao_);
  glBindBuffer(GL_ARRAY_BUFFER, quadVbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
  glBindVertexArray(0);

  return true;
}

bool Renderer::EnsureFramebuffers(int viewportWidth, int viewportHeight) {
  if (viewportWidth <= 0 || viewportHeight <= 0) {
    return false;
  }

  if (viewportWidth_ == viewportWidth && viewportHeight_ == viewportHeight && hdrFbo_ != 0) {
    return true;
  }

  viewportWidth_ = viewportWidth;
  viewportHeight_ = viewportHeight;

  if (hdrColorTex_ != 0) glDeleteTextures(1, &hdrColorTex_);
  if (hdrBrightTex_ != 0) glDeleteTextures(1, &hdrBrightTex_);
  if (hdrDepthRbo_ != 0) glDeleteRenderbuffers(1, &hdrDepthRbo_);
  if (hdrFbo_ == 0) glGenFramebuffers(1, &hdrFbo_);

  glBindFramebuffer(GL_FRAMEBUFFER, hdrFbo_);

  glGenTextures(1, &hdrColorTex_);
  glBindTexture(GL_TEXTURE_2D, hdrColorTex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, viewportWidth_, viewportHeight_, 0, GL_RGBA, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdrColorTex_, 0);

  glGenTextures(1, &hdrBrightTex_);
  glBindTexture(GL_TEXTURE_2D, hdrBrightTex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, viewportWidth_, viewportHeight_, 0, GL_RGBA, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, hdrBrightTex_, 0);

  const GLenum attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, attachments);

  glGenRenderbuffers(1, &hdrDepthRbo_);
  glBindRenderbuffer(GL_RENDERBUFFER, hdrDepthRbo_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, viewportWidth_, viewportHeight_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, hdrDepthRbo_);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "HDR framebuffer is incomplete\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return false;
  }

  if (pingPongTex_[0] != 0) glDeleteTextures(2, pingPongTex_);
  if (pingPongFbo_[0] == 0) glGenFramebuffers(2, pingPongFbo_);
  glGenTextures(2, pingPongTex_);

  for (int i = 0; i < 2; ++i) {
    glBindFramebuffer(GL_FRAMEBUFFER, pingPongFbo_[i]);
    glBindTexture(GL_TEXTURE_2D, pingPongTex_[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, viewportWidth_, viewportHeight_, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingPongTex_[i], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      std::cerr << "Ping-pong framebuffer is incomplete\n";
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      return false;
    }
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  return true;
}

bool Renderer::EnsureShadowResources() {
  if (shadowFbo_ != 0 && shadowDepthTex_ != 0) {
    return true;
  }

  glGenFramebuffers(1, &shadowFbo_);
  glGenTextures(1, &shadowDepthTex_);

  glBindTexture(GL_TEXTURE_2D, shadowDepthTex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, settings_.shadowResolution, settings_.shadowResolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  const float border[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

  glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowDepthTex_, 0);
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);

  const bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  if (!ok) {
    std::cerr << "Shadow framebuffer is incomplete\n";
    return false;
  }
  return true;
}

void Renderer::DrawMeshByType(GeometryType type) const {
  if (type == GeometryType::Cuboid) {
    cuboid_.Draw();
  } else if (type == GeometryType::Cylinder) {
    cylinder_.Draw();
  } else {
    prism_.Draw();
  }
}

void Renderer::RenderSceneGeometry(GLuint program, const std::vector<Component>& components, std::uint32_t selectedId, bool drawFloor, bool drawSelectionOverlay) {
  if (drawFloor) {
    const auto floorIt = MaterialCatalog().find(MaterialType::Grass);
    if (floorIt != MaterialCatalog().end()) {
      EnsureMaterialTexturesLoaded(MaterialType::Grass, MutableMaterialCatalog().at(MaterialType::Grass));
      const glm::mat4 floorModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.001f, 0.0f)) *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(220.0f, 1.0f, 220.0f));
      const glm::mat3 floorNormal = glm::inverseTranspose(glm::mat3(floorModel));

      if (program == pbrProgram_) {
        const MaterialDefinition& floorMat = MaterialCatalog().at(MaterialType::Grass);
        glDisable(GL_CULL_FACE);
        BindMaterialTextures(program, floorMat);
        glUniform1f(glGetUniformLocation(program, "uUvScale"), 120.0f);
      }

      glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 1, GL_FALSE, glm::value_ptr(floorModel));
      if (program == pbrProgram_) {
        glUniformMatrix3fv(glGetUniformLocation(program, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(floorNormal));
      }
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

    glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
    if (program == pbrProgram_) {
      glUniformMatrix3fv(glGetUniformLocation(program, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(normalMat));
      glUniform1f(glGetUniformLocation(program, "uUvScale"), 1.0f);
      glUniform1f(glGetUniformLocation(program, "uGlobalRoughnessMul"), settings_.globalRoughnessMultiplier * mat.roughnessMultiplier);
      glUniform1f(glGetUniformLocation(program, "uGlobalMetallicMul"), settings_.globalMetallicMultiplier * mat.metallicMultiplier);
      BindMaterialTextures(program, mat);
    }

    DrawMeshByType(c.geometry);

    if (drawSelectionOverlay && program == pbrProgram_ && c.id == selectedId) {
      glEnable(GL_POLYGON_OFFSET_LINE);
      glPolygonOffset(-1.0f, -1.0f);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glDisable(GL_CULL_FACE);
      glLineWidth(2.0f);

      glUniform1i(glGetUniformLocation(program, "uUseUnlitColor"), 1);
      glUniform3f(glGetUniformLocation(program, "uUnlitColor"), 1.0f, 0.85f, 0.2f);
      DrawMeshByType(c.geometry);

      glUniform1i(glGetUniformLocation(program, "uUseUnlitColor"), 0);
      glLineWidth(1.0f);
      glEnable(GL_CULL_FACE);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glDisable(GL_POLYGON_OFFSET_LINE);
    }
  }
}

bool Renderer::Initialize() {
  if (!BuildPrograms()) {
    return false;
  }
  if (!InitializeFullscreenQuad()) {
    return false;
  }

  InitializeMaterialCatalog();
  StartTextureWorker();

  for (auto& [type, mat] : MutableMaterialCatalog()) {
    EnsureMaterialTexturesLoaded(type, mat);
  }

  WaitForAllTextureUploadsAtStartup();

  cuboid_ = GpuMesh(CreateCuboidMesh());
  cylinder_ = GpuMesh(CreateCylinderMesh(28));
  prism_ = GpuMesh(CreatePrismMesh());
  plane_ = GpuMesh(CreatePlaneMesh());

  CreateSimpleEnvironmentCubemap(envCubemap_);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  return true;
}

void Renderer::Shutdown() {
  StopTextureWorker();
  ReleaseMaterialTextures();

  DeleteIfValid(pbrProgram_, true);
  DeleteIfValid(depthProgram_, true);
  DeleteIfValid(blurProgram_, true);
  DeleteIfValid(postProgram_, true);

  if (quadVbo_ != 0) {
    glDeleteBuffers(1, &quadVbo_);
    quadVbo_ = 0;
  }
  if (quadVao_ != 0) {
    glDeleteVertexArrays(1, &quadVao_);
    quadVao_ = 0;
  }

  if (hdrColorTex_ != 0) glDeleteTextures(1, &hdrColorTex_);
  if (hdrBrightTex_ != 0) glDeleteTextures(1, &hdrBrightTex_);
  if (hdrDepthRbo_ != 0) glDeleteRenderbuffers(1, &hdrDepthRbo_);
  if (hdrFbo_ != 0) glDeleteFramebuffers(1, &hdrFbo_);
  hdrColorTex_ = hdrBrightTex_ = hdrDepthRbo_ = hdrFbo_ = 0;

  if (pingPongTex_[0] != 0) glDeleteTextures(2, pingPongTex_);
  if (pingPongFbo_[0] != 0) glDeleteFramebuffers(2, pingPongFbo_);
  pingPongTex_[0] = pingPongTex_[1] = 0;
  pingPongFbo_[0] = pingPongFbo_[1] = 0;

  if (shadowDepthTex_ != 0) glDeleteTextures(1, &shadowDepthTex_);
  if (shadowFbo_ != 0) glDeleteFramebuffers(1, &shadowFbo_);
  shadowDepthTex_ = shadowFbo_ = 0;

  if (envCubemap_ != 0) glDeleteTextures(1, &envCubemap_);
  envCubemap_ = 0;
}

void Renderer::DrawScene(const std::vector<Component>& components, std::uint32_t selectedId, const OrbitCamera& camera, int viewportWidth, int viewportHeight, bool drawFloor) {
  if (viewportWidth <= 0 || viewportHeight <= 0) {
    return;
  }

  if (!EnsureFramebuffers(viewportWidth, viewportHeight)) {
    return;
  }
  if (!EnsureShadowResources()) {
    return;
  }

  PumpDecodedTextureUploads(6);

  const glm::mat4 view = camera.ViewMatrix();
  const glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight), 0.1f, 500.0f);
  const glm::vec3 camPos = camera.Position();
  const glm::mat4 lightViewProj = ComputeDirectionalLightMatrix(settings_);

  if (settings_.enableShadows) {
    glViewport(0, 0, settings_.shadowResolution, settings_.shadowResolution);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo_);
    glClear(GL_DEPTH_BUFFER_BIT);

    glUseProgram(depthProgram_);
    glUniformMatrix4fv(glGetUniformLocation(depthProgram_, "uLightViewProj"), 1, GL_FALSE, glm::value_ptr(lightViewProj));
    RenderSceneGeometry(depthProgram_, components, selectedId, drawFloor, false);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, hdrFbo_);
  glViewport(0, 0, viewportWidth, viewportHeight);
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.06f, 0.09f, 0.14f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(pbrProgram_);
  glUniformMatrix4fv(glGetUniformLocation(pbrProgram_, "uView"), 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(pbrProgram_, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
  glUniformMatrix4fv(glGetUniformLocation(pbrProgram_, "uLightViewProj"), 1, GL_FALSE, glm::value_ptr(lightViewProj));
  glUniform3fv(glGetUniformLocation(pbrProgram_, "uCameraPos"), 1, glm::value_ptr(camPos));

  glUniform3fv(glGetUniformLocation(pbrProgram_, "uPointLightPos"), 1, glm::value_ptr(settings_.pointLightPosition));
  glUniform3fv(glGetUniformLocation(pbrProgram_, "uPointLightColor"), 1, glm::value_ptr(settings_.pointLightColor));
  glUniform1f(glGetUniformLocation(pbrProgram_, "uPointLightIntensity"), settings_.pointLightIntensity);

  glUniform3fv(glGetUniformLocation(pbrProgram_, "uDirLightDir"), 1, glm::value_ptr(settings_.directionalDirection));
  glUniform3fv(glGetUniformLocation(pbrProgram_, "uDirLightColor"), 1, glm::value_ptr(settings_.directionalColor));
  glUniform1f(glGetUniformLocation(pbrProgram_, "uDirLightIntensity"), settings_.directionalIntensity);

  glUniform3fv(glGetUniformLocation(pbrProgram_, "uSkyAmbientColor"), 1, glm::value_ptr(settings_.skyAmbientColor));
  glUniform3fv(glGetUniformLocation(pbrProgram_, "uGroundAmbientColor"), 1, glm::value_ptr(settings_.groundAmbientColor));
  glUniform1f(glGetUniformLocation(pbrProgram_, "uAmbientIntensity"), settings_.ambientIntensity);

  glUniform1f(glGetUniformLocation(pbrProgram_, "uAoStrength"), settings_.aoStrength);
  glUniform1f(glGetUniformLocation(pbrProgram_, "uParallaxHeightScale"), settings_.parallaxHeightScale);
  glUniform1i(glGetUniformLocation(pbrProgram_, "uEnableShadows"), settings_.enableShadows ? 1 : 0);
  glUniform1i(glGetUniformLocation(pbrProgram_, "uShadowPcfRadius"), settings_.shadowPcfRadius);
  glUniform1i(glGetUniformLocation(pbrProgram_, "uUseUnlitColor"), 0);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap_);
  glUniform1i(glGetUniformLocation(pbrProgram_, "envMap"), 6);

  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D, shadowDepthTex_);
  glUniform1i(glGetUniformLocation(pbrProgram_, "shadowMap"), 7);

  RenderSceneGeometry(pbrProgram_, components, selectedId, drawFloor, true);

  bool horizontal = true;
  bool firstIteration = true;
  const int blurPasses = settings_.enableBloom ? 8 : 0;

  glUseProgram(blurProgram_);
  for (int i = 0; i < blurPasses; ++i) {
    glBindFramebuffer(GL_FRAMEBUFFER, pingPongFbo_[horizontal ? 1 : 0]);
    glUniform1i(glGetUniformLocation(blurProgram_, "uHorizontal"), horizontal ? 1 : 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, firstIteration ? hdrBrightTex_ : pingPongTex_[horizontal ? 0 : 1]);
    glUniform1i(glGetUniformLocation(blurProgram_, "uImage"), 0);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(quadVao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    horizontal = !horizontal;
    if (firstIteration) firstIteration = false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, viewportWidth, viewportHeight);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(postProgram_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, hdrColorTex_);
  glUniform1i(glGetUniformLocation(postProgram_, "uScene"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, settings_.enableBloom ? pingPongTex_[horizontal ? 0 : 1] : hdrBrightTex_);
  glUniform1i(glGetUniformLocation(postProgram_, "uBloom"), 1);

  glUniform1f(glGetUniformLocation(postProgram_, "uExposure"), settings_.exposure);
  glUniform1f(glGetUniformLocation(postProgram_, "uBloomStrength"), settings_.bloomStrength);
  glUniform1i(glGetUniformLocation(postProgram_, "uEnableBloom"), settings_.enableBloom ? 1 : 0);
  glUniform1i(glGetUniformLocation(postProgram_, "uEnableFxaa"), settings_.enableFxaa ? 1 : 0);
  glUniform2f(glGetUniformLocation(postProgram_, "uInvViewport"), 1.0f / static_cast<float>(viewportWidth), 1.0f / static_cast<float>(viewportHeight));

  glBindVertexArray(quadVao_);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);

  glEnable(GL_DEPTH_TEST);
}
