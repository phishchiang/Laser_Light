  
export const sceneUniformConfig = {
    viewMatrix: {
        size: 16,          // Size of the view matrix (4x4 matrix)
        offset: 0,         // Offset for view matrix in the uniform buffer
    },
    projectionMatrix: {
        size: 16,          // Size of the projection matrix (4x4 matrix)
        offset: 16,        // Offset for projection matrix in the uniform buffer
    },
    canvasSize: {
        size: 2,           // Size of the canvas size (vec2)
        offset: 32,        // Offset for canvas size in the uniform buffer
    },
    uTime: {
        size: 1,           // Size of the time value (float)
        offset: 34,        // Offset for uTime in the uniform buffer
    },
};

// Object-level uniforms
export const objectUniformConfig = {
    modelMatrix: {
        size: 16,          // Size of the model matrix (4x4 matrix)
        offset: 0,         // Offset for model matrix in the uniform buffer
    },
    uTestValue: {
        size: 1,           // Size of the test value (float)
        offset: 16,        // Offset for uTestValue in the uniform buffer
    },
    uTestValue_02: {
        size: 1,           // Size of the test value (float)
        offset: 17,        // Offset for uTestValue_02 in the uniform buffer
    },
    uLightIntensity: {
        size: 1,
        offset: 18,
    },
    uLightColor: {
        size: 4,
        offset: 20,
    },
};