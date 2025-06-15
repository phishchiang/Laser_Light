export interface PipelineBuilderOptions {
  vertexShaderCode: string;
  fragmentShaderCode: string;
  vertexEntryPoint?: string;
  fragmentEntryPoint?: string;
  vertexLayout?: GPUVertexBufferLayout;
  bindGroupLayouts: GPUBindGroupLayout[];
  targets: GPUColorTargetState[];
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
}

export class PipelineBuilder {
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  public createPipeline(options: PipelineBuilderOptions): GPURenderPipeline {
    const {
      vertexShaderCode,
      fragmentShaderCode,
      vertexEntryPoint = 'vertex_main',
      fragmentEntryPoint = 'fragment_main',
      vertexLayout,
      bindGroupLayouts,
      targets,
      primitive = { topology: 'triangle-list', cullMode: 'none' },
      depthStencil,
    } = options;

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts,
    });

    return this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: this.device.createShaderModule({ code: vertexShaderCode }),
        entryPoint: vertexEntryPoint,
        ...(vertexLayout ? { buffers: [vertexLayout] } : {}),
      },
      fragment: {
        module: this.device.createShaderModule({ code: fragmentShaderCode }),
        entryPoint: fragmentEntryPoint,
        targets,
      },
      primitive,
      ...(depthStencil ? { depthStencil } : {}),
    });
  }
}