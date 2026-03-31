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
