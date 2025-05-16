export class getVertexLayout {
    private attributes: GPUVertexAttribute[] = [];
    private offset: number = 0;
  
    constructor(private vertexFormat: { position: boolean; normal?: boolean; color?: boolean; uv?: boolean }) {}
  
    build(): { arrayStride: number; attributes: GPUVertexAttribute[] } {
      this.attributes = [];
      this.offset = 0;
  
      if (this.vertexFormat.position) this.addAttribute(0, 'float32x3', 3); // Position
      if (this.vertexFormat.normal) this.addAttribute(1, 'float32x3', 3); // Normal
      if (this.vertexFormat.color) this.addAttribute(2, 'float32x4', 4); // Color
      if (this.vertexFormat.uv) this.addAttribute(3, 'float32x2', 2); // UV
  
      return {
        arrayStride: this.offset,
        attributes: this.attributes,
      };
    }
  
    private addAttribute(shaderLocation: number, format: GPUVertexFormat, componentCount: number) {
      this.attributes.push({
        shaderLocation,
        offset: this.offset,
        format,
      });
      this.offset += componentCount * 4; // 4 bytes per float
    }
  }
  