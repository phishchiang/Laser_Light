export class PipelineBuilder {
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  public createPipeline(
    presentationFormat: GPUTextureFormat,
    vertexShaderCode: string,
    fragmentShaderCode: string,
    vertexLayout: GPUVertexBufferLayout,
    blend?: GPUBlendState 
  ): {
    pipeline: GPURenderPipeline;
    sceneBindGroupLayout: GPUBindGroupLayout;
    objectBindGroupLayout: GPUBindGroupLayout;
  } {
      // Create the scene-level bind group layout
      const sceneBindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0, // Scene-level uniforms
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          },
        ],
      });

      // Create the object-level bind group layout
      const objectBindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0, // Object-level uniforms
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          },
          {
            binding: 1, // Sampler
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: 'filtering' },
          },
          {
            binding: 2, // Texture
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'float' },
          },
        ],
      });

      // Create the pipeline layout with both bind group layouts
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [sceneBindGroupLayout, objectBindGroupLayout],
      });

      // Create the render pipeline
      const pipeline = this.device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: this.device.createShaderModule({
            code: vertexShaderCode,
          }),
          entryPoint: 'vertex_main',
          buffers: [vertexLayout],
        },
        fragment: {
          module: this.device.createShaderModule({
            code: fragmentShaderCode,
          }),
          entryPoint: 'fragment_main',
          targets: [
            {
              format: presentationFormat,
              blend: blend,
            },
          ],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',
        },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less',
        },
      });

    return { pipeline, sceneBindGroupLayout, objectBindGroupLayout };
  }
}