// Scene-level uniforms
struct SceneUniforms {
  viewMatrix : mat4x4<f32>,
  projectionMatrix : mat4x4<f32>,
  canvasSize : vec2f,
  uTime : f32,
};

// Object-level uniforms
struct ObjectUniforms {
  modelMatrix : mat4x4<f32>,
  uTestValue : f32,
  uTestValue_02 : f32,
};

@group(0) @binding(0) var<uniform> sceneUniforms : SceneUniforms;
@group(1) @binding(0) var<uniform> objectUniforms : ObjectUniforms;
@group(1) @binding(1) var mySampler: sampler;
@group(1) @binding(2) var myTexture: texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(3) uv : vec2f,
}

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec3f,
  @location(1) frag_worldPosition : vec3f,
  @location(2) frag_uv : vec2f,
}

// Hash function for pseudo-random gradients
fn hash3(p: vec3f) -> f32 {
  let p3 = fract(p * 0.1031);
  let p4 = dot(p3, p3.yzx + 19.19);
  return fract((p4 + 19.19) * p4);
}

// Simple value noise in 3D
fn valueNoise3D(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);

  // Trilinear interpolation of 8 corners
  let c000 = hash3(i + vec3f(0.0, 0.0, 0.0));
  let c100 = hash3(i + vec3f(1.0, 0.0, 0.0));
  let c010 = hash3(i + vec3f(0.0, 1.0, 0.0));
  let c110 = hash3(i + vec3f(1.0, 1.0, 0.0));
  let c001 = hash3(i + vec3f(0.0, 0.0, 1.0));
  let c101 = hash3(i + vec3f(1.0, 0.0, 1.0));
  let c011 = hash3(i + vec3f(0.0, 1.0, 1.0));
  let c111 = hash3(i + vec3f(1.0, 1.0, 1.0));

  let u = f * f * (3.0 - 2.0 * f); // Smoothstep

  let x00 = mix(c000, c100, u.x);
  let x10 = mix(c010, c110, u.x);
  let x01 = mix(c001, c101, u.x);
  let x11 = mix(c011, c111, u.x);

  let y0 = mix(x00, x10, u.y);
  let y1 = mix(x01, x11, u.y);

  return mix(y0, y1, u.z);
}

// Add a Direction Array for Each Octave
const octaveDirs = array<vec3f, 5>(
  vec3f(1.0, 0.5, 0.0),
  vec3f(-0.7, 1.0, 0.2),
  vec3f(0.3, -0.6, 1.0),
  vec3f(-1.0, 0.2, -0.5),
  vec3f(0.5, -1.0, 0.7)
);

// fBM (fractional Brownian motion)
fn fbm(p: vec3f) -> f32 {
  var value = 0.0;
  var amplitude = 0.5;
  var frequency = 1.0;
  for (var i = 0; i < 5; i = i + 1) {
    // Animate each octave in a different direction and speed
    let timeOffset = sceneUniforms.uTime * (0.2 + 0.15 * f32(i));
    let animatedP = p * frequency + octaveDirs[i] * timeOffset;
    value = value + amplitude * valueNoise3D(animatedP);
    frequency = frequency * 2.0;
    amplitude = amplitude * 0.5;
  }
  return value;
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
  let translateYMatrix = mat4x4<f32>(
    1.0, 0.0, 0.0, 0.0,  // Scale X by 1.0
    0.0, 1.0, 0.0, 0.0,  // Scale Y by 1.0
    0.0, 0.0, 1.0, 0.0,  // Scale Z by 1.0
    0.0, 0.0, objectUniforms.uTestValue_02, 1.0   
  );

  var transformedModelMatrix = objectUniforms.modelMatrix * translateYMatrix;
  let worldPosition = (transformedModelMatrix * vec4f(input.position, 1.0)).xyz;

  return VertexOutput(
    sceneUniforms.projectionMatrix * sceneUniforms.viewMatrix * transformedModelMatrix * vec4f(input.position, 1.0), 
    input.normal,
    worldPosition,
    input.uv,
  );
}

struct FragmentInput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec3f,
  @location(1) frag_worldPosition : vec3f,
  @location(2) frag_uv : vec2f,
}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
  // let noisePos = vec3f(input.frag_uv*5.0, sceneUniforms.uTime * 0.5); // Use the UV from GLB

  let center = vec2f(0.5, 1.0);
  let dist = distance(input.frag_uv, center);
  var invertVignette = smoothstep(0.0, 1.2, dist);
  invertVignette = pow(invertVignette, 4.0); // Adjust the power to control vignette strength

  // Animate along Z for turbulence
  let noisePos = input.frag_worldPosition * 5.0 + vec3f(0.0, sceneUniforms.uTime * 0.5, sceneUniforms.uTime * 0.16);
  var noiseValue = fbm(noisePos);

  // apply power to noise value
  noiseValue = pow(noiseValue, 1.0); // Uncomment to apply power to the noise value

  var UVGradientMul = noiseValue * (1.0 - input.frag_uv.y);
  // UVGradientMul = UVGradientMul + (invertVignette * UVGradientMul);
  UVGradientMul = UVGradientMul + invertVignette * objectUniforms.uTestValue;

  var finalColor: vec4f = textureSample(myTexture, mySampler, input.frag_uv);
  // finalColor = vec4f(input.frag_worldPosition.x, input.frag_worldPosition.y, input.frag_worldPosition.z, 1.0);
  finalColor = vec4f(UVGradientMul, UVGradientMul, UVGradientMul, UVGradientMul);
  return finalColor;
}