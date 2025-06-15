import { mat4, vec3 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import edgeLightWGSL from './edgeLight.wgsl?raw'; // Raw String Import but only specific to Vite.
import triNoiseWGSL from './triNoise.wgsl?raw'; 
import { ArcballCamera, WASDCamera } from './camera';
import { createInputHandler } from './input';
import { loadAndProcessGLB } from './loadAndProcessGLB';
import { sceneUniformConfig, objectUniformConfig } from './uniformConfig';
import { PipelineBuilder } from './PipelineBuilder';
import { PipelineBuilderOptions } from './PipelineBuilder';
import { RenderTarget } from './RenderTarget';
import { PostProcessEffect } from './postprocessing/PostProcessEffect';
import { PassThroughEffect } from './postprocessing/PassThroughEffect';
import { GrayscaleEffect } from './postprocessing/GrayscaleEffect';
import { FXAAEffect } from './postprocessing/FXAAEffect';
// Glow FX imports
import { BrightPassEffect } from './postprocessing/BrightPassEffect';
import { BlurEffect } from './postprocessing/GaussianBlurEffect';
import { AddEffect } from './postprocessing/AddEffect';

const MESH_PATH = '/assets/meshes/lightEdge.glb';
const LIGHT_TRI_PATH = '/assets/meshes/lightTri.glb';

export class WebGPUApp{
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private triPipeline!: GPURenderPipeline;
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
    all_translate_X: number;
    all_translate_Y: number;
    all_translate_Z: number;
    all_rotate_X: number;
    all_rotate_Y: number;
    all_rotate_Z: number;
    uLightIntensity: number; 
    uLightColor: [number, number, number]; 
    u_p1_X: number;
    u_p1_Y: number;
    u_p1_Z: number;
    u_p2_X: number;
    u_p2_Y: number;
    u_p2_Z: number;
    u_p3_X: number;
    u_p3_Y: number;
    u_p3_Z: number;
    uGlow_Threshold: number;
    uGlow_Radius: number;
    uGlow_Intensity: number;
  } = {
    type: 'arcball',
    uTestValue: 1.0,
    uTestValue_02: 1.0,
    all_translate_X: 0,
    all_translate_Y: 0,
    all_translate_Z: 0,
    all_rotate_X: 0,
    all_rotate_Y: 0,
    all_rotate_Z: 0,
    uLightIntensity: 1.0,
    uLightColor: [0.106, 0.686, 0.776], // 27, 175, 198
    u_p1_X: 0.0,
    u_p1_Y: 0.5,
    u_p1_Z: 0.0,
    u_p2_X: -1,
    u_p2_Y: -1,
    u_p2_Z: 0.0,
    u_p3_X: 1,
    u_p3_Y: -1,
    u_p3_Z: 0.0,
    uGlow_Threshold: 0.15,
    uGlow_Radius: 15.0,
    uGlow_Intensity: 0.08,
  };
  private uTime: number = 0.0;
  private gui: GUI;
  private lastFrameMS: number;
  private demoVerticesBuffer!: GPUBuffer;
  private sideLine_01_VerticesBuffer!: GPUBuffer;
  private sideLine_02_VerticesBuffer!: GPUBuffer;
  private loadIndexBuffer!: GPUBuffer | undefined;
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
  private sideLine_01_interVertexData!: Float32Array;
  private sideLine_02_interVertexData!: Float32Array;
  private tri_interVertexData!: Float32Array;
  private tri_VerticesBuffer!: GPUBuffer;
  private tri_IndexBuffer!: GPUBuffer | undefined;
  private tri_IndexCount!: number;
  private tri_VertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[] };
  private inputHandler!: () => { 
    digital: { forward: boolean, backward: boolean, left: boolean, right: boolean, up: boolean, down: boolean, };
    analog: { x: number; y: number; zoom: number; touching: boolean };
  };
  private renderTarget_ping!: RenderTarget;
  private renderTarget_pong!: RenderTarget;
  private postProcessEffects: PostProcessEffect[] = [];
  private passThroughEffect!: PassThroughEffect;
  // Glow FX Variables
  private renderTarget_baseOut_glow!: RenderTarget;
  private renderTarget_ping_glow!: RenderTarget;
  private renderTarget_pong_glow!: RenderTarget;
  private brightPassEffect!: BrightPassEffect;
  private blurEffectH!: BlurEffect;
  private blurEffectV!: BlurEffect;
  private addEffect!: AddEffect;
  private enableGlow: boolean = true; // or control with GUI
  private static readonly CLEAR_COLOR = [0.1, 0.1, 0.1, 1.0];
  private static readonly CAMERA_POSITION = vec3.create(0.5, 0, 3);

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
    this.initRenderTargetsForPP();
    await this.initLoadAndProcessGLB();
    this.initUniformBuffer();
    await this.loadTexture();
    this.initCam();
    this.initPipelineBindGrp();
    this.initializeGUI();
    this.setupEventListeners();
    this.renderFrame();
  }

  private initRenderTargetsForPP() {
    // Create ping-pong render targets
    this.renderTarget_ping = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );
    this.renderTarget_pong = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );

    // Init 3 render targets for glow bloom effect
    this.renderTarget_baseOut_glow = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );
    this.renderTarget_ping_glow = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );
    this.renderTarget_pong_glow = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );

    // Init useful pass-through effect 
    this.passThroughEffect = new PassThroughEffect(this.device, this.presentationFormat, this.sampler);

    // Add post-processing effects
    this.postProcessEffects.push(
      // new GrayscaleEffect(this.device, this.presentationFormat, this.sampler),
      new FXAAEffect(this.device, this.presentationFormat, this.sampler, [this.canvas.width, this.canvas.height]),
    );

    this.brightPassEffect = new BrightPassEffect(this.device, this.presentationFormat, this.sampler, this.params.uGlow_Threshold );
    this.blurEffectH = new BlurEffect(this.device, this.presentationFormat, this.sampler, [1.0, 0.0], [1 / this.canvas.width, 1 / this.canvas.height], this.params.uGlow_Radius );
    this.blurEffectV = new BlurEffect(this.device, this.presentationFormat, this.sampler, [0.0, 1.0], [1 / this.canvas.width, 1 / this.canvas.height], this.params.uGlow_Radius );
    this.addEffect = new AddEffect(this.device, this.presentationFormat, this.sampler, this.params.uGlow_Intensity );
  }

  private getGlobalTransformMatrix(): Float32Array {
    // Start with identity
    let m = mat4.identity();

    // Helper to convert degrees to radians
    const deg2rad = (deg: number) => deg * Math.PI / 180;

    // Apply rotations (order: X, Y, Z)
    m = mat4.rotateX(m, deg2rad(this.params.all_rotate_X));
    m = mat4.rotateY(m, deg2rad(this.params.all_rotate_Y));
    m = mat4.rotateZ(m, deg2rad(this.params.all_rotate_Z));

    // Apply translation
    m = mat4.translate(m, [
      this.params.all_translate_X,
      this.params.all_translate_Y,
      this.params.all_translate_Z
    ]);

    return m;
  }

  private transformPoint(point: Float32Array, matrix: Float32Array): Float32Array {
    return vec3.transformMat4(point, matrix);
  }

  private updateEdgeVertices_01() {
    // Get GUI points
    let p1 = vec3.create(this.params.u_p1_X, this.params.u_p1_Y, this.params.u_p1_Z);
    let p2 = vec3.create(this.params.u_p2_X, this.params.u_p2_Y, this.params.u_p2_Z);

    // Apply global transform
    const globalMatrix = this.getGlobalTransformMatrix();
    p1 = this.transformPoint(p1, globalMatrix);
    p2 = this.transformPoint(p2, globalMatrix);

    // Compute direction and right vector for width
    const upDir = vec3.normalize(vec3.subtract(p2, p1));
    // Use camera position to get a normal
    const cameraPos = this.cameras[this.params.type].position;
    const center = vec3.scale(vec3.add(p1, p2), 0.5);
    const toCamera = vec3.normalize(vec3.subtract(cameraPos, center));
    const right = vec3.normalize(vec3.cross(upDir, toCamera));

    // Set half width (adjust to your mesh's original half-width)
    const halfWidth = 0.02;

    // Set the tiny bias value to shift the upper vertices slightly up
    const upperBias = 0.01;

    // Calculate four corners
    // Top edge (p1)
    let leftUp = vec3.add(p1, vec3.scale(right, -halfWidth));
    let rightUp = vec3.add(p1, vec3.scale(right, halfWidth));
    // Bottom edge (p2)
    const leftDown = vec3.add(p2, vec3.scale(right, -halfWidth));
    const rightDown = vec3.add(p2, vec3.scale(right, halfWidth));

    // Adjust the upper vertices slightly up
    leftUp = vec3.add(leftUp, vec3.scale(upDir, -upperBias));
    rightUp = vec3.add(rightUp, vec3.scale(upDir, -upperBias));

    const vertexOrder = [rightUp, leftUp, leftDown, rightUp, leftDown, rightDown];

    const stride = this.loadVertexLayout.arrayStride / 4; // floats per vertex
    // console.log(this.loadVertexLayout)
    for (let i = 0; i < 6; i++) {
      this.sideLine_01_interVertexData[i * stride + 0] = vertexOrder[i][0];
      this.sideLine_01_interVertexData[i * stride + 1] = vertexOrder[i][1];
      this.sideLine_01_interVertexData[i * stride + 2] = vertexOrder[i][2];
    }

    this.device.queue.writeBuffer( this.sideLine_01_VerticesBuffer, 0, this.sideLine_01_interVertexData.buffer, 0, this.sideLine_01_interVertexData.byteLength );
  }

  private updateEdgeVertices_02() {
    let p1 = vec3.create(this.params.u_p1_X, this.params.u_p1_Y, this.params.u_p1_Z);
    let p3 = vec3.create(this.params.u_p3_X, this.params.u_p3_Y, this.params.u_p3_Z);

    // Apply global transform
    const globalMatrix = this.getGlobalTransformMatrix();
    p1 = this.transformPoint(p1, globalMatrix);
    p3 = this.transformPoint(p3, globalMatrix);

    const upDir = vec3.normalize(vec3.subtract(p3, p1));
    const cameraPos = this.cameras[this.params.type].position;
    const center = vec3.scale(vec3.add(p1, p3), 0.5);
    const toCamera = vec3.normalize(vec3.subtract(cameraPos, center));
    const right = vec3.normalize(vec3.cross(upDir, toCamera));
    const halfWidth = 0.02;

    // Set the tiny bias value to shift the upper vertices slightly up
    const upperBias = 0.01;

    let leftUp = vec3.add(p1, vec3.scale(right, -halfWidth));
    let rightUp = vec3.add(p1, vec3.scale(right, halfWidth));
    const leftDown = vec3.add(p3, vec3.scale(right, -halfWidth));
    const rightDown = vec3.add(p3, vec3.scale(right, halfWidth));

    // Adjust the upper vertices slightly up
    leftUp = vec3.add(leftUp, vec3.scale(upDir, -upperBias));
    rightUp = vec3.add(rightUp, vec3.scale(upDir, -upperBias));

    const vertexOrder = [rightUp, leftUp, leftDown, rightUp, leftDown, rightDown];
    const stride = this.loadVertexLayout.arrayStride / 4;

    for (let i = 0; i < 6; i++) {
      this.sideLine_02_interVertexData[i * stride + 0] = vertexOrder[i][0];
      this.sideLine_02_interVertexData[i * stride + 1] = vertexOrder[i][1];
      this.sideLine_02_interVertexData[i * stride + 2] = vertexOrder[i][2];
    }

    this.device.queue.writeBuffer(
      this.sideLine_02_VerticesBuffer,
      0,
      this.sideLine_02_interVertexData.buffer,
      0,
      this.sideLine_02_interVertexData.byteLength
    );
  }

  private updateTriVertices() {
    // Get GUI points
    let p1 = vec3.create(this.params.u_p1_X, this.params.u_p1_Y, this.params.u_p1_Z); // top
    let p2 = vec3.create(this.params.u_p2_X, this.params.u_p2_Y, this.params.u_p2_Z); // bottom left
    let p3 = vec3.create(this.params.u_p3_X, this.params.u_p3_Y, this.params.u_p3_Z); // bottom right

    // Apply global transform
    const globalMatrix = this.getGlobalTransformMatrix();
    p1 = this.transformPoint(p1, globalMatrix);
    p2 = this.transformPoint(p2, globalMatrix);
    p3 = this.transformPoint(p3, globalMatrix);

    // The order here must match your GLB's original vertex order!
    // For a typical triangle: [top, bottom left, bottom right]
    const positions = [p2, p3, p1];

    const stride = this.tri_VertexLayout.arrayStride / 4; // floats per vertex

    for (let i = 0; i < 3; i++) {
      this.tri_interVertexData[i * stride + 0] = positions[i][0];
      this.tri_interVertexData[i * stride + 1] = positions[i][1];
      this.tri_interVertexData[i * stride + 2] = positions[i][2];
    }

    // Upload to GPU
    this.device.queue.writeBuffer(
      this.tri_VerticesBuffer,
      0,
      this.tri_interVertexData.buffer,
      0,
      this.tri_interVertexData.byteLength
    );
  }

  private async initLoadAndProcessGLB() {
    const {   
      interleavedData: edgeInterleavedData,
      indices: edgeIndexData,
      indexCount: edgeIndexCount,
      vertexLayout: edgeVertexLayout
    } = await loadAndProcessGLB( MESH_PATH);

    // For mesh 1
    this.sideLine_01_interVertexData = new Float32Array(edgeInterleavedData); // Make a copy
    this.sideLine_01_VerticesBuffer = this.device.createBuffer({
      size: this.sideLine_01_interVertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.sideLine_01_VerticesBuffer,
      0,
      this.sideLine_01_interVertexData.buffer,
      0,
      this.sideLine_01_interVertexData.byteLength
    );

    // For mesh 2
    this.sideLine_02_interVertexData = new Float32Array(edgeInterleavedData); // Make a copy
    this.sideLine_02_VerticesBuffer = this.device.createBuffer({
      size: this.sideLine_02_interVertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.sideLine_02_VerticesBuffer,
      0,
      this.sideLine_02_interVertexData.buffer,
      0,
      this.sideLine_02_interVertexData.byteLength
    );

    // Create index buffer if indices exist
    let indexBuffer: GPUBuffer | undefined = undefined;
    if (edgeIndexData) {
      // Create index buffer
      // Pad index buffer size to next multiple of 4 for avoiding alignment issues
      // WebGPU requires buffer sizes to be a multiple of 4 bytes
      const paddedIndexBufferSize = Math.ceil(edgeIndexData.byteLength / 4) * 4;

      indexBuffer = this.device.createBuffer({
        size: paddedIndexBufferSize,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint16Array(indexBuffer.getMappedRange()).set(edgeIndexData);
      indexBuffer.unmap();
    }

    this.loadIndexBuffer = indexBuffer;
    this.loadIndexCount = edgeIndexCount;
    this.loadVertexLayout = edgeVertexLayout;

    // Load the triangle mesh
    const {
      interleavedData: triInterleavedData,
      indices: triIndexData,
      indexCount: triIndexCount,
      vertexLayout: triVertexLayout
    } = await loadAndProcessGLB(LIGHT_TRI_PATH);

    this.tri_interVertexData = new Float32Array(triInterleavedData); // Make a copy
    this.tri_VerticesBuffer = this.device.createBuffer({
      size: this.tri_interVertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.tri_VerticesBuffer,
      0,
      this.tri_interVertexData.buffer,
      0,
      this.tri_interVertexData.byteLength
    );

    if (triIndexData) {
      // Create index buffer
      // Pad index buffer size to next multiple of 4 for avoiding alignment issues
      // WebGPU requires buffer sizes to be a multiple of 4 bytes
      const paddedIndexBufferSize = Math.ceil(triIndexData.byteLength / 4) * 4;

      indexBuffer = this.device.createBuffer({
        size: paddedIndexBufferSize,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint16Array(indexBuffer.getMappedRange()).set(triIndexData);
      indexBuffer.unmap();
    }

    this.tri_IndexBuffer = indexBuffer;
    this.tri_IndexCount = triIndexCount;
    this.tri_VertexLayout = triVertexLayout;

    this.updateEdgeVertices_01();
    this.updateEdgeVertices_02();
    this.updateTriVertices();
  }

  private initCam(){
    this.aspect = this.canvas.width / this.canvas.height;
    this.projectionMatrix = mat4.perspective((2 * Math.PI) / 5, this.aspect, 0.1, 100.0);
    
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
    objectUniformData.set([this.params.uLightIntensity], objectUniformConfig.uLightIntensity.offset);
    objectUniformData.set(this.params.uLightColor, objectUniformConfig.uLightColor.offset);

    // Write data to the uniform buffer
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, sceneUniformData.buffer, 0, sceneUniformData.byteLength);
    this.device.queue.writeBuffer(this.objectUniformBuffer, 0, objectUniformData.buffer, 0, objectUniformData.byteLength);

    console.log('Object Uniform Buffer:', objectUniformData);
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

    // Resize the render targets
    this.renderTarget_ping.resize(this.device, this.canvas.width, this.canvas.height, this.presentationFormat);
    this.renderTarget_pong.resize(this.device, this.canvas.width, this.canvas.height, this.presentationFormat);
  }

  private initializeGUI() {
    
    // this.gui.add(this.params, 'uTestValue', 0.0, 1.0).step(0.01).onChange((value) => {
    //   this.updateFloatUniform( 'uTestValue', value );
    // });
    // this.gui.add(this.params, 'uTestValue_02', 0.0, 1.0).step(0.01).onChange((value) => {
    //   this.updateFloatUniform( 'uTestValue_02', value );
    // });

    const u_allFolder = this.gui.addFolder('All Points Transform');
    u_allFolder.open();

    u_allFolder.add(this.params, 'all_translate_X', -3, 3).step(0.001).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_allFolder.add(this.params, 'all_translate_Y', -3, 3).step(0.001).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_allFolder.add(this.params, 'all_translate_Z', -3, 3).step(0.001).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_allFolder.add(this.params, 'all_rotate_X', -180, 180).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_allFolder.add(this.params, 'all_rotate_Y', -180, 180).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_allFolder.add(this.params, 'all_rotate_Z', -180, 180).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    
    u_allFolder.add(this.params, 'uLightIntensity', 0.0, 1.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uLightIntensity', value );
    });

    // Add color picker
    u_allFolder.addColor(this.params, 'uLightColor').onChange((value) => {
      const normalizedColor = value.map((v: number) => v / 255.0);
      const colorArray = new Float32Array(normalizedColor);
      this.device.queue.writeBuffer( this.objectUniformBuffer, objectUniformConfig.uLightColor.offset * 4, colorArray.buffer, 0, colorArray.byteLength );
    });

    const u_p1Folder = this.gui.addFolder('1st Point Position');
    // u_p1Folder.open();

    u_p1Folder.add(this.params, 'u_p1_X', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_p1Folder.add(this.params, 'u_p1_Y', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_p1Folder.add(this.params, 'u_p1_Z', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });

    const u_p2Folder = this.gui.addFolder('2nd Point Position');
    // u_p2Folder.open();

    u_p2Folder.add(this.params, 'u_p2_X', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateTriVertices();
    });
    u_p2Folder.add(this.params, 'u_p2_Y', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateTriVertices();
    });
    u_p2Folder.add(this.params, 'u_p2_Z', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_01();
      this.updateTriVertices();
    });

    const u_p3Folder = this.gui.addFolder('3rd Point Position');
    // u_p3Folder.open();

    u_p3Folder.add(this.params, 'u_p3_X', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_p3Folder.add(this.params, 'u_p3_Y', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });
    u_p3Folder.add(this.params, 'u_p3_Z', -10, 10).step(0.01).onChange(() => {
      this.updateEdgeVertices_02();
      this.updateTriVertices();
    });

    const glowFolder = this.gui.addFolder('Glow FX');
    glowFolder.add(this.params, 'uGlow_Threshold', 0.0, 1.0).step(0.01).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_Radius', 0.1, 20.0).step(0.1).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_Intensity', 0.0, 1.0).step(0.01).onChange(() => this.updateGlowUniforms());
    glowFolder.open();
    
  }

  private updateGlowUniforms() {
    this.brightPassEffect.setThreshold(this.params.uGlow_Threshold);
    this.blurEffectH.setRadius(this.params.uGlow_Radius);
    this.blurEffectV.setRadius(this.params.uGlow_Radius);
    this.addEffect.setIntensity(this.params.uGlow_Intensity);
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
      case 'uLightIntensity':
        offset = objectUniformConfig.uLightIntensity.offset * 4;
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


  private initPipelineBindGrp() {
    const pipelineBuilder = new PipelineBuilder(this.device);

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


    const edgeOptions: PipelineBuilderOptions = {
      vertexShaderCode: edgeLightWGSL,
      fragmentShaderCode: edgeLightWGSL,
      vertexEntryPoint: 'vertex_main',
      fragmentEntryPoint: 'fragment_main',
      vertexLayout: {
        arrayStride: this.loadVertexLayout.arrayStride,
        attributes: this.loadVertexLayout.attributes,
      },
      bindGroupLayouts: [sceneBindGroupLayout, objectBindGroupLayout],
      targets: [{
        format: this.presentationFormat,
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one',
            operation: 'add',
          },
        },
        writeMask: GPUColorWrite.ALL,
      }],
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    };

    this.pipeline = pipelineBuilder.createPipeline(edgeOptions);

    const triOptions: PipelineBuilderOptions = {
      vertexShaderCode: triNoiseWGSL,
      fragmentShaderCode: triNoiseWGSL,
      vertexEntryPoint: 'vertex_main',
      fragmentEntryPoint: 'fragment_main',
      vertexLayout: {
        arrayStride: this.tri_VertexLayout.arrayStride,
        attributes: this.tri_VertexLayout.attributes,
      },
      bindGroupLayouts: [sceneBindGroupLayout, objectBindGroupLayout],
      targets: [{
        format: this.presentationFormat,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
        writeMask: GPUColorWrite.ALL,
      }],
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    };

    this.triPipeline = pipelineBuilder.createPipeline(triOptions);

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

  private applyGlowBloom(commandEncoder: GPUCommandEncoder, lastPPOutputView: GPUTextureView) {
    // Store the last post-processed output in the base render target for glow bloom
    this.passThroughEffect.apply(
      commandEncoder,
      { A: lastPPOutputView },
      this.renderTarget_baseOut_glow.view,
      [this.canvas.width, this.canvas.height]
    );
  
    // 1. Bright pass: extract brights from the scene
    this.brightPassEffect.apply(
      commandEncoder,
      { A: this.renderTarget_baseOut_glow.view },
      this.renderTarget_ping_glow.view,
      // this.context.getCurrentTexture().createView(),
      [this.canvas.width, this.canvas.height]
    );

    // 2. Blur the bright pass
    this.blurEffectH.apply(
      commandEncoder,
      { A: this.renderTarget_ping_glow.view },
      this.renderTarget_pong_glow.view,
      // this.context.getCurrentTexture().createView(),
      [this.canvas.width, this.canvas.height]
    );

    // 3. Blur the result (vertical)
    this.blurEffectV.apply(
      commandEncoder,
      { A: this.renderTarget_pong_glow.view },
      this.renderTarget_ping_glow.view,
      [this.canvas.width, this.canvas.height]
    );

    // 4. Add blurred brights back to the scene
    this.addEffect.apply(
      commandEncoder,
      { A: this.renderTarget_baseOut_glow.view, B: this.renderTarget_ping_glow.view},
      this.context.getCurrentTexture().createView(),
      [this.canvas.width, this.canvas.height]
    );
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

    // Set up a render pass target based on post-processing effects
    if (this.postProcessEffects.length === 0) {
      (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context.getCurrentTexture().createView();
    } else {
      (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.renderTarget_ping.view;
    }

    // Update the depth attachment view
    this.renderPassDescriptor.depthStencilAttachment!.view = this.depthTexture.createView();

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    // Draw the 1st edge mesh
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.sceneUniformBindGroup); // Scene-level uniforms
    passEncoder.setBindGroup(1, this.objectUniformBindGroup); // Object-level uniforms

    passEncoder.setVertexBuffer(0, this.sideLine_01_VerticesBuffer);
    passEncoder.setIndexBuffer(this.loadIndexBuffer!, 'uint16');
    passEncoder.draw(this.loadIndexCount);

    // Draw the triangle mesh
    passEncoder.setPipeline(this.triPipeline);
    passEncoder.setBindGroup(0, this.sceneUniformBindGroup); // Scene-level uniforms
    passEncoder.setBindGroup(1, this.objectUniformBindGroup); // Object-level uniforms
    passEncoder.setVertexBuffer(0, this.tri_VerticesBuffer);
    passEncoder.setIndexBuffer(this.tri_IndexBuffer!, 'uint16');
    passEncoder.draw(this.tri_IndexCount);

    // Draw the 2nd edge mesh
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setVertexBuffer(0, this.sideLine_02_VerticesBuffer);
    passEncoder.setIndexBuffer(this.loadIndexBuffer!, 'uint16');
    passEncoder.draw(this.loadIndexCount);

    passEncoder.end();

    // Apply post-processing effects if any
    let finalOutputView = this.renderTarget_ping.view;
    if (this.postProcessEffects.length > 0) {
      let inputView = this.renderTarget_ping.view;
      let outputView = this.renderTarget_pong.view;
      for (let i = 0; i < this.postProcessEffects.length; i++) {
        const isLast = i === this.postProcessEffects.length - 1;
        // finalOutputView = isLast ? this.context.getCurrentTexture().createView() : outputView; // Only use single output for PostProcessEffects
        finalOutputView = outputView; // Make sure to continue using ping-pong buffers when applying glowFX afterwards
        this.postProcessEffects[i].apply(
          commandEncoder,
          { A: inputView },
          finalOutputView,
          [this.canvas.width, this.canvas.height]
        );
        if (!isLast) {
          [inputView, outputView] = [outputView, inputView];
        }
      }
      if (this.enableGlow) {
        this.applyGlowBloom(commandEncoder, finalOutputView);
      }
    }

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.renderFrame.bind(this));
  }
}

const app = new WebGPUApp(document.getElementById('app') as HTMLCanvasElement);
