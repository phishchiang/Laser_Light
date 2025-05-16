import { mat4 } from 'wgpu-matrix';
import { uniformConfig } from './uniformConfig';
import { uniformConfigScene } from './uniformConfigScene';

export class Scene {
  private device: GPUDevice;
  private uniformBuffer: GPUBuffer;
  private viewMatrix: Float32Array;
  private projectionMatrix: Float32Array;
  private canvasSize: Float32Array;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;

    // Initialize matrices
    this.viewMatrix = mat4.identity();
    this.projectionMatrix = mat4.identity();
    this.canvasSize = new Float32Array([canvas.width, canvas.height]);

    // Create the uniform buffer
    const uniformBufferSize = Math.ceil((4 * 48) / 16) * 16; // Ensure alignment
    this.uniformBuffer = this.device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initialize the projection matrix
    this.updateProjectionMatrix(canvas.width / canvas.height);
  }

  public updateProjectionMatrix(aspect: number) {
    this.projectionMatrix = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      uniformConfig.projectionMatrix.offset * 4,
      this.projectionMatrix.buffer,
      this.projectionMatrix.byteOffset,
      this.projectionMatrix.byteLength
    );
  }

  public updateViewMatrix(viewMatrix: Float32Array) {
    this.viewMatrix = viewMatrix;
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      uniformConfig.viewMatrix.offset * 4,
      this.viewMatrix.buffer,
      this.viewMatrix.byteOffset,
      this.viewMatrix.byteLength
    );
  }

  public updateCanvasSize(width: number, height: number) {
    this.canvasSize = new Float32Array([width, height]);
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      uniformConfig.canvasSize.offset * 4,
      this.canvasSize.buffer,
      0,
      this.canvasSize.byteLength
    );
  }

  public getUniformBuffer(): GPUBuffer {
    return this.uniformBuffer;
  }
}