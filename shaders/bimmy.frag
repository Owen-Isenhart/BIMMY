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
  vec2 p = viewDir.xy / max(viewDir.z, 0.08) * ((h - 0.5) * uParallaxHeightScale);
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
  BrightColor = (luminance > 1.0) ? vec4(color, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}
