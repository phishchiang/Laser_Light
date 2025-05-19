import { mat4, vec3 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import basicWGSL from './basic.wgsl?raw'; // Raw String Import but only specific to Vite.
import { ArcballCamera, WASDCamera } from './camera';
import { createInputHandler } from './input';
import { loadAndProcessGLB } from './loadAndProcessGLB';
import { sceneUniformConfig, objectUniformConfig } from './uniformConfig';
import { PipelineBuilder } from './PipelineBuilder';

const MESH_PATH = '/assets/meshes/lightEdge.glb';

export class WebGPUApp{
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private presentationFormat!: GPUTextureFormat;
  private uniformBindGroup!: GPUBindGroup;
  private sceneUniformBindGroup!: GPUBindGroup;
  private objectUniformBindGroup!: GPUBindGroup;
  private renderPassDescriptor!: GPURenderPassDescriptor;
  private cubeTexture!: GPUTexture;
  private cameras: { [key: string]: any };
  private aspect!: number;
  private params: { 
    type: 'arcball' | 'WASD'; 
    uTestValue: number; 
    uTestValue_02: number; 
    u_p1_X: number;
    u_p1_Y: number;
    u_p1_Z: number;
    u_p2_X: number;
    u_p2_Y: number;
    u_p2_Z: number;
    u_p3_X: number;
    u_p3_Y: number;
    u_p3_Z: number;
  } = {
    type: 'arcball',
    uTestValue: 1.0,
    uTestValue_02: 1.0,
    u_p1_X: 0.0,
    u_p1_Y: 0.0,
    u_p1_Z: 0.0,
    u_p2_X: -1,
    u_p2_Y: -1,
    u_p2_Z: 0.0,
    u_p3_X: 1,
    u_p3_Y: -1,
    u_p3_Z: 0.0,
  };
  private uTime: number = 0.0;
  private gui: GUI;
  private lastFrameMS: number;
  private demoVerticesBuffer!: GPUBuffer;
  private loadVerticesBuffer!: GPUBuffer;
  private loadIndexBuffer!: GPUBuffer;
  private loadIndexCount!: number;
  private uniformBuffer!: GPUBuffer;
  private sceneUniformBuffer!: GPUBuffer;
  private objectUniformBuffer!: GPUBuffer;
  private loadVertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[]; };
  private modelMatrix: Float32Array;
  private viewMatrix: Float32Array;
  private projectionMatrix: Float32Array;
  private depthTexture!: GPUTexture;
  private sampler!: GPUSampler;
  private newCameraType!: string;
  private oldCameraType!: string;
  private interleavedVertexData!: Float32Array;
  private inputHandler!: () => { 
    digital: { forward: boolean, backward: boolean, left: boolean, right: boolean, up: boolean, down: boolean, };
    analog: { x: number; y: number; zoom: number; touching: boolean };
  };
  private static readonly CLEAR_COLOR = [0.1, 0.1, 0.1, 1.0];
  private static readonly CAMERA_POSITION = vec3.create(3, 2, 5);

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.gui = new GUI();
    this.cameras = {
      arcball: new ArcballCamera({ position: WebGPUApp.CAMERA_POSITION }),
      WASD: new WASDCamera({ position: WebGPUApp.CAMERA_POSITION }),
    };
    this.oldCameraType = this.params.type;
    this.lastFrameMS = Date.now();
    this.sampler = {} as GPUSampler;

     // The input handler
    this.inputHandler = createInputHandler(window, this.canvas);

    // Initialize matrices
    this.modelMatrix = mat4.identity();
    this.viewMatrix = mat4.identity();
    this.projectionMatrix = mat4.identity();

    this.setupAndRender();
  }

  public async setupAndRender() {
    await this.initializeWebGPU();
    await this.initLoadAndProcessGLB();
    this.initUniformBuffer();
    await this.loadTexture();
    this.initCam();
    this.initPipelineBindGrp(this.presentationFormat);
    this.initializeGUI();
    this.setupEventListeners();
    this.renderFrame();
  }

  private updateEdgeVertices() {
    // Get GUI points
    const p1 = vec3.create(this.params.u_p1_X, this.params.u_p1_Y, this.params.u_p1_Z);
    const p2 = vec3.create(this.params.u_p2_X, this.params.u_p2_Y, this.params.u_p2_Z);

    // Compute direction and right vector for width
    const upDir = vec3.normalize(vec3.subtract(p2, p1));
    // Use camera position to get a normal
    const cameraPos = this.cameras[this.params.type].position;
    const center = vec3.scale(vec3.add(p1, p2), 0.5);
    const toCamera = vec3.normalize(vec3.subtract(cameraPos, center));
    const right = vec3.normalize(vec3.cross(upDir, toCamera));

    // Set half width (adjust to your mesh's original half-width)
    const halfWidth = 0.05;

    // Calculate four corners
    // Top edge (p1)
    const leftUp = vec3.add(p1, vec3.scale(right, -halfWidth));
    const rightUp = vec3.add(p1, vec3.scale(right, halfWidth));
    // Bottom edge (p2)
    const leftDown = vec3.add(p2, vec3.scale(right, -halfWidth));
    const rightDown = vec3.add(p2, vec3.scale(right, halfWidth));

    const vertexOrder = [rightUp, leftUp, leftDown, rightUp, leftDown, rightDown];

    const stride = this.loadVertexLayout.arrayStride / 4; // floats per vertex
    console.log(this.loadVertexLayout)
    for (let i = 0; i < 6; i++) {
      this.interleavedVertexData[i * stride + 0] = vertexOrder[i][0];
      this.interleavedVertexData[i * stride + 1] = vertexOrder[i][1];
      this.interleavedVertexData[i * stride + 2] = vertexOrder[i][2];
    }

    this.device.queue.writeBuffer( this.loadVerticesBuffer, 0, this.interleavedVertexData.buffer, 0, this.interleavedVertexData.byteLength );
  }

  private async initLoadAndProcessGLB() {
    const { vertexBuffer, indexBuffer, indexCount, vertexLayout, interleavedData } = await loadAndProcessGLB(this.device, MESH_PATH);
  
    this.loadVerticesBuffer = vertexBuffer;
    this.loadIndexBuffer = indexBuffer;
    this.loadIndexCount = indexCount;
    this.loadVertexLayout = vertexLayout;
    this.interleavedVertexData = interleavedData;
    console.log(vertexBuffer);
  }

  private initCam(){
    this.aspect = this.canvas.width / this.canvas.height;
    this.projectionMatrix = mat4.perspective((2 * Math.PI) / 5, this.aspect, 1, 100.0);
    
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;
    // Projection mat4x4<f32> 16 value 64 bytes, index 32
    this.device.queue.writeBuffer(this.sceneUniformBuffer, sceneUniformConfig.projectionMatrix.offset * 4, this.projectionMatrix.buffer, this.projectionMatrix.byteOffset, this.projectionMatrix.byteLength); // Projection matrix
  }

  private async loadTexture() {
    const response = await fetch('../assets/img/uv1.png');
    const imageBitmap = await createImageBitmap(await response.blob());

    this.cubeTexture = this.device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: this.cubeTexture },
      [imageBitmap.width, imageBitmap.height]
    );
  }

  private initUniformBuffer() {
    // Calculate the total size for scene-level uniforms
    const sceneUniformTotalSize = Object.values(sceneUniformConfig)
    .reduce((sum, item) => sum + item.size, 0);

    // Calculate the total size for object-level uniforms
    const objectUniformTotalSize = Object.values(objectUniformConfig)
    .reduce((sum, item) => sum + item.size, 0);

    // Ensure sizes are aligned to 16 bytes
    const sceneUniformBufferSize = Math.ceil((sceneUniformTotalSize * 4) / 16) * 16;
    const objectUniformBufferSize = Math.ceil((objectUniformTotalSize * 4) / 16) * 16;

    // Create buffers
    this.sceneUniformBuffer = this.device.createBuffer({
      size: sceneUniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    this.objectUniformBuffer = this.device.createBuffer({
      size: objectUniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  

    // Prepare data for scene-level uniforms
    const sceneUniformData = new Float32Array(sceneUniformBufferSize / 4);
    sceneUniformData.set(this.viewMatrix, sceneUniformConfig.viewMatrix.offset);
    sceneUniformData.set(this.projectionMatrix, sceneUniformConfig.projectionMatrix.offset);
    sceneUniformData.set([this.canvas.width, this.canvas.height], sceneUniformConfig.canvasSize.offset);
    sceneUniformData.set([this.uTime], sceneUniformConfig.uTime.offset);

    // Prepare data for object-level uniforms
    const objectUniformData = new Float32Array(objectUniformBufferSize / 4);
    objectUniformData.set(this.modelMatrix, objectUniformConfig.modelMatrix.offset);
    objectUniformData.set([this.params.uTestValue], objectUniformConfig.uTestValue.offset);
    objectUniformData.set([this.params.uTestValue_02], objectUniformConfig.uTestValue_02.offset);

    // Write data to the uniform buffer
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, sceneUniformData.buffer, 0, sceneUniformData.byteLength);
    this.device.queue.writeBuffer(this.objectUniformBuffer, 0, objectUniformData.buffer, 0, objectUniformData.byteLength);

    // console.log('Object Uniform Buffer:', objectUniformData);
  }

  private setupEventListeners() {
    window.addEventListener('resize', this.resize.bind(this));
  }


  private resize() {
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.aspect = this.canvas.width / this.canvas.height;
    this.projectionMatrix = mat4.perspective((2 * Math.PI) / 5, this.aspect, 1, 100.0);
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
    });
    // Projection mat4x4<f32> 16 value 64 bytes, index 32
    this.device.queue.writeBuffer(this.sceneUniformBuffer, sceneUniformConfig.projectionMatrix.offset * 4, this.projectionMatrix.buffer, this.projectionMatrix.byteOffset, this.projectionMatrix.byteLength); 

    // CanvasSize vec2f 2 value 8 bytes, index 48
    const canvasSizeArray = new Float32Array([this.canvas.width, this.canvas.height]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, sceneUniformConfig.canvasSize.offset * 4, canvasSizeArray.buffer, 0, canvasSizeArray.byteLength);

    // Recreate the depth texture to match the new canvas size
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private initializeGUI() {
    
    this.gui.add(this.params, 'uTestValue', 0.0, 1.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uTestValue', value );
    });
    this.gui.add(this.params, 'uTestValue_02', 0.0, 1.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uTestValue_02', value );
    });

    const u_p1Folder = this.gui.addFolder('1st Point Position');
    u_p1Folder.open();

    u_p1Folder.add(this.params, 'u_p1_X', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p1_X', value );
      this.updateEdgeVertices();
    });
    u_p1Folder.add(this.params, 'u_p1_Y', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p1_Y', value );
      this.updateEdgeVertices();
    });
    u_p1Folder.add(this.params, 'u_p1_Z', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p1_Z', value );
      this.updateEdgeVertices();
    });

    const u_p2Folder = this.gui.addFolder('2nd Point Position');
    u_p2Folder.open();

    u_p2Folder.add(this.params, 'u_p2_X', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p2_X', value );
      this.updateEdgeVertices();
    });
    u_p2Folder.add(this.params, 'u_p2_Y', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p2_Y', value );
      this.updateEdgeVertices();
    });
    u_p2Folder.add(this.params, 'u_p2_Z', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p2_Z', value );
      this.updateEdgeVertices();
    });

    const u_p3Folder = this.gui.addFolder('3rd Point Position');
    u_p3Folder.open();

    u_p3Folder.add(this.params, 'u_p3_X', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p3_X', value );
      this.updateEdgeVertices();
    });
    u_p3Folder.add(this.params, 'u_p3_Y', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p3_Y', value );
      this.updateEdgeVertices();
    });
    u_p3Folder.add(this.params, 'u_p3_Z', -10, 10).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'u_p3_Z', value );
      this.updateEdgeVertices();
    });
    
  }

  private updateFloatUniform(key: keyof typeof this.params, value: number) {
    let offset: number = 0;
    switch (key) {
      case 'uTestValue':
        offset = objectUniformConfig.uTestValue.offset * 4;
        break;
      case 'uTestValue_02':
        offset = objectUniformConfig.uTestValue_02.offset * 4;;
        break;
      case 'u_p1_X':
        offset = objectUniformConfig.u_p1_X.offset * 4;
        break;
      case 'u_p1_Y':
        offset = objectUniformConfig.u_p1_Y.offset * 4;
        break;
      case 'u_p1_Z':
        offset = objectUniformConfig.u_p1_Z.offset * 4;
        break;
      case 'u_p2_X':
        offset = objectUniformConfig.u_p2_X.offset * 4;
        break;
      case 'u_p2_Y':
        offset = objectUniformConfig.u_p2_Y.offset * 4;
        break;
      case 'u_p2_Z':
        offset = objectUniformConfig.u_p2_Z.offset * 4;
        break;
      case 'u_p3_X':
        offset = objectUniformConfig.u_p3_X.offset * 4;
        break;
      case 'u_p3_Y':
        offset = objectUniformConfig.u_p3_Y.offset * 4;
        break;
      case 'u_p3_Z':
        offset = objectUniformConfig.u_p3_Z.offset * 4;
        break;
      default:
        console.error(`Unknown key: ${key}`);
        return;
    }

    const updatedFloatArray = new Float32Array([value]);
    this.device.queue.writeBuffer(this.objectUniformBuffer, offset, updatedFloatArray.buffer, 0, updatedFloatArray.byteLength);
  }

  private async initializeWebGPU() {
    const adapter = await navigator.gpu?.requestAdapter({ featureLevel: 'compatibility' });
    this.device = await adapter?.requestDevice() as GPUDevice;

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    });

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });


    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: undefined, // Assigned later
          clearValue: WebGPUApp.CLEAR_COLOR,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ] as Iterable< GPURenderPassColorAttachment | null | undefined>,
      depthStencilAttachment: {
        view: this.depthTexture.createView(), // Assign a valid GPUTextureView
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };
  }


  private initPipelineBindGrp(presentationFormat: GPUTextureFormat) {
    const pipelineBuilder = new PipelineBuilder(this.device);

    const { pipeline, sceneBindGroupLayout, objectBindGroupLayout } = pipelineBuilder.createPipeline(
      presentationFormat,
      basicWGSL,
      basicWGSL,
      {
          arrayStride: this.loadVertexLayout.arrayStride,
          attributes: this.loadVertexLayout.attributes,
      }
    );

    this.pipeline = pipeline;

    // Create the scene-level uniform bind group
    this.sceneUniformBindGroup = this.device.createBindGroup({
      layout: sceneBindGroupLayout,
      entries: [
        {
          binding: 0, // Scene-level uniforms
          resource: { buffer: this.sceneUniformBuffer },
        },
      ],
    });

    // Create the object-level uniform bind group
    this.objectUniformBindGroup = this.device.createBindGroup({
      layout: objectBindGroupLayout,
      entries: [
        {
          binding: 0, // Object-level uniforms
          resource: { buffer: this.objectUniformBuffer },
        },
        {
          binding: 1, // Sampler
          resource: this.sampler,
        },
        {
          binding: 2, // Texture
          resource: this.cubeTexture.createView(),
        },
      ],
    });
  }

  private getViewMatrix(deltaTime: number) {
    const camera = this.cameras[this.params.type];
    const viewMatrix =  camera.update(deltaTime, this.inputHandler());
    return viewMatrix;
  }

  private renderFrame() {
    const now = Date.now();
    const deltaTime = (now - this.lastFrameMS) / 1000;
    this.lastFrameMS = now;

    // Update the uniform uTime value
    this.uTime += deltaTime;
    const uTimeFloatArray = new Float32Array([this.uTime]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, sceneUniformConfig.uTime.offset * 4, uTimeFloatArray.buffer, 0, uTimeFloatArray.byteLength);

    this.viewMatrix = this.getViewMatrix(deltaTime);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, sceneUniformConfig.viewMatrix.offset * 4, this.viewMatrix.buffer, 0, this.viewMatrix.byteLength);

    (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context
    .getCurrentTexture()
    .createView();

    // Update the depth attachment view
    this.renderPassDescriptor.depthStencilAttachment!.view = this.depthTexture.createView();

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    // passEncoder.setBindGroup(0, this.uniformBindGroup);
    passEncoder.setBindGroup(0, this.sceneUniformBindGroup); // Scene-level uniforms
    passEncoder.setBindGroup(1, this.objectUniformBindGroup); // Object-level uniforms
    passEncoder.setVertexBuffer(0, this.loadVerticesBuffer);
    passEncoder.setIndexBuffer(this.loadIndexBuffer, 'uint16');
    passEncoder.draw(this.loadIndexCount);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.renderFrame.bind(this));
  }
}

const app = new WebGPUApp(document.getElementById('app') as HTMLCanvasElement);
