import { mat4 } from 'wgpu-matrix';

export class Object3D {
  private device: GPUDevice;
  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private indexCount: number;
  private objUniformBuffer: GPUBuffer;
  private modelMatrix: Float32Array;
  private texture: GPUTexture;
  private sampler: GPUSampler;

  constructor(
    device: GPUDevice,
    geometry: { vertices: GPUBuffer; indices: GPUBuffer; indexCount: number; vertexLayout: { arrayStride: number; attributes: GPUVertexAttribute[] } },
    material: { texture?: GPUTexture; sampler?: GPUSampler } = {}
  ) {
    this.device = device;
  
    // Assign geometry buffers
    this.vertexBuffer = geometry.vertices;
    this.indexBuffer = geometry.indices;
    this.indexCount = geometry.indexCount;
  
    // Initialize the model matrix to the identity matrix
    this.modelMatrix = mat4.identity();
  
    // Create uniform buffer for the model matrix
    this.objUniformBuffer = this.device.createBuffer({
      size: 64, // 4x4 matrix (16 floats * 4 bytes each)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  
    // Initialize the uniform buffer with the model matrix
    this.updateobjUniformBuffer();
  
    // Assign material properties
    this.texture = material.texture || this.createDefaultTexture();
    this.sampler = material.sampler || this.createDefaultSampler();
  }

  private createDefaultTexture(): GPUTexture {
    const texture = this.device.createTexture({
      size: [1, 1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Fill the texture with white color
    const whitePixel = new Uint8Array([255, 255, 255, 255]);
    this.device.queue.writeTexture(
      { texture },
      whitePixel,
      { bytesPerRow: 4 },
      [1, 1, 1]
    );

    return texture;
  }

  private createDefaultSampler(): GPUSampler {
    return this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  public updateModelMatrix(matrix: Float32Array) {
    this.modelMatrix = matrix;
    this.updateobjUniformBuffer();
  }

  private updateobjUniformBuffer() {
    this.device.queue.writeBuffer(
      this.objUniformBuffer,
      0,
      this.modelMatrix.buffer,
      this.modelMatrix.byteOffset,
      this.modelMatrix.byteLength
    );
  }

  public render(
    passEncoder: GPURenderPassEncoder,
    pipeline: GPURenderPipeline,
    sceneBindGroup: GPUBindGroup // Accept scene-level bind group as a parameter
  ) {
    // Create a bind group for this object (object-level uniforms, texture, sampler)
    const objectBindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1), // Object-level bind group layout
      entries: [
        {
          binding: 0, // Uniform buffer for the model matrix
          resource: { buffer: this.objUniformBuffer },
        },
        {
          binding: 1, // Sampler
          resource: this.sampler,
        },
        {
          binding: 2, // Texture
          resource: this.texture.createView(),
        },
      ],
    });

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, sceneBindGroup); // Bind the scene-level bind group
    passEncoder.setBindGroup(1, objectBindGroup); // Bind the object-level bind group
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint16');
    passEncoder.drawIndexed(this.indexCount);
  }
}