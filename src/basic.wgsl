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
  @location(2) frag_uv : vec2f,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
  let translateYMatrix = mat4x4<f32>(
    1.0, 0.0, 0.0, 0.0,  // Scale X by 1.0
    0.0, 1.0, 0.0, 0.0,  // Scale Y by 1.0
    0.0, 0.0, 1.0, 0.0,  // Scale Z by 1.0
    0.0, 0.0, objectUniforms.uTestValue_02, 1.0   // Translation along Y-axis
  );

  var transformedModelMatrix = objectUniforms.modelMatrix * translateYMatrix;

  return VertexOutput(
    sceneUniforms.projectionMatrix * sceneUniforms.viewMatrix * transformedModelMatrix * vec4f(input.position, 1.0), 
    input.normal,
    input.uv,
  );
}

struct FragmentInput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec3f,
  @location(2) frag_uv : vec2f,
}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
  // var finalColor: vec4f = textureSample(myTexture, mySampler, input.frag_uv);

  var UVGradient = pow(1.0 - input.frag_uv.y, 1.5);
  let centerV = input.frag_uv.x;
  let centerFade = 1.0 - smoothstep(0.1, 0.5, abs(centerV - 0.5));
  UVGradient = UVGradient * centerFade;

  var finalColor = vec4f(UVGradient, UVGradient, UVGradient, UVGradient);
  return finalColor;
}