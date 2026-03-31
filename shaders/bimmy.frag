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

  vec3 n = GetNormalFromMap(uv);
  vec3 v = normalize(uCameraPos - vWorldPos);
  vec3 l = normalize(lightPos - vWorldPos);
  vec3 h = normalize(v + l);

  float distance = length(lightPos - vWorldPos);
  float attenuation = 1.0 / max(distance * distance, 0.0001);
  vec3 radiance = lightColor * attenuation;

  vec3 f0 = vec3(0.04);
  f0 = mix(f0, albedo, metallic);

  float ndf = DistributionGGX(n, h, roughness);
  float g = GeometrySmith(n, v, l, roughness);
  vec3 f = FresnelSchlick(max(dot(h, v), 0.0), f0);

  vec3 numerator = ndf * g * f;
  float denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0);
  vec3 specular = numerator / max(denominator, 0.0001);

  vec3 kS = f;
  vec3 kD = vec3(1.0) - kS;
  kD *= (1.0 - metallic);

  float nDotL = max(dot(n, l), 0.0);
  vec3 lo = (kD * albedo / PI + specular) * radiance * nDotL;

  vec3 ambient = vec3(0.03) * albedo * ao;
  vec3 color = ambient + lo;

  color = color / (color + vec3(1.0));
  color = pow(color, vec3(1.0 / 2.2));

  FragColor = vec4(color, 1.0);
}
