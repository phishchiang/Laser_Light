export const uniformConfigScene = {         
    viewMatrix:{
        size: 16,          // Size of the view matrix (4x4 matrix)
        offset: 16,        // Offset for view matrix in the uniform buffer
    },
    projectionMatrix:{
        size: 16,          // Size of the projection matrix (4x4 matrix)
        offset: 32,        // Offset for projection matrix in the uniform buffer
    },  
    canvasSize: {
        size: 2,            // Size of the canvas size (vec2)
        offset: 48,         // Offset for canvas size in the uniform buffer
    },
  };