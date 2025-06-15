import { PostProcessEffect } from './PostProcessEffect';
import passThroughWGSL from '../shaders/passThrough.wgsl?raw';
import { PipelineBuilder, PipelineBuilderOptions } from '../PipelineBuilder';

export class PassThroughEffect implements PostProcessEffect {
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
      vertexShaderCode: passThroughWGSL,
      fragmentShaderCode: passThroughWGSL,
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
    
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: outputView,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputView },
      ],
    });

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.draw(6, 1, 0, 0); // Fullscreen quad
    passEncoder.end();
  }
}