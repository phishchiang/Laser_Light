import grayscaleWGSL from '../shaders/grayscale.wgsl?raw';
import { PostProcessEffect } from './PostProcessEffect';
import { PipelineBuilder, PipelineBuilderOptions } from '../PipelineBuilder';


export class GrayscaleEffect implements PostProcessEffect {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private pipelineBuilder: PipelineBuilder;

  constructor(device: GPUDevice, format: GPUTextureFormat, sampler: GPUSampler) {
    this.device = device;
    this.sampler = sampler;
    this.pipelineBuilder = new PipelineBuilder(device);
    
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });

    const options: PipelineBuilderOptions = {
      vertexShaderCode: grayscaleWGSL,
      fragmentShaderCode: grayscaleWGSL,
      vertexEntryPoint: 'vs_main',
      fragmentEntryPoint: 'fs_main',
      bindGroupLayouts: [this.bindGroupLayout],
      targets: [{ format }],
      primitive: { topology: 'triangle-list' },
      // No depthStencil for post-process
    };

    this.pipeline = this.pipelineBuilder.createPipeline(options);
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView },
    outputView: GPUTextureView,
    _size: [number, number]
  ): void {
    const inputView = input.A; // Use the 'A' key for the main input

    // Recreate bind group if inputView changes
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputView },
      ],
    });

    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: outputView,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const pass = commandEncoder.beginRenderPass(passDesc);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }
}