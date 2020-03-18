
#include <optix.h>
#include "LaunchParams.h" // our launch params
#include <vec_math.h> // NVIDIAs math utils


extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}
// ray type -> these must be in the 
// same order as the ray types in Nau's project
enum { PHONG_RAY_TYPE=0, SHADOW, RAY_TYPE_COUNT };



// -------------------------------------------------------
// closest hit computes color based on material color or texture

extern "C" __global__ void __closesthit__radiance()
{

    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    float4 LightDir = optixLaunchParams.global->lightDir;
    
    // get the payload variable
    float3 &prd = *(float3*)getPRD<float3>();

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // get mesh data
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        prd= make_float3(fromTexture);
    
    }
    else
        prd = sbtData.color;

    // Normal do pixel
    float4 normal = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];


    // I=L•N*C*I_l
    float intensidade = max(dot( -normalize(LightDir),normalize(normal)),0.f) ;


    float4 pos
        = (1.f-u-v) * sbtData.vertexD.position[index.x]
        +         u * sbtData.vertexD.position[index.y]
        +         v * sbtData.vertexD.position[index.z];

    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                make_float3(-LightDir),
                0.001f, // tmin
                1e20f, // tmax
                0.0f, // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                SHADOW, // SBT offset
                RAY_TYPE_COUNT, // SBT stride
                SHADOW, // missSBTIndex
                u0, u1 );   


    const auto &camera = optixLaunchParams.camera; 
    
    float3 specularColor = make_float3(0,0,0);

    float3 vertexToEye = normalize(make_float3(pos)-camera.position);
    float3 lightReflect = make_float3(normalize(reflect(-LightDir,normal)));
    float specularFactor = dot(vertexToEye,lightReflect);
    if (specularFactor > 0) {
        int shininess = 1;
        specularFactor = pow(specularFactor,shininess);
        specularColor = pixelColorPRD * 1 * specularFactor;
    }

    // Componente ambiente + componente difusa*intensidade + componente especular
    prd = (prd*0.2 + prd*pixelColorPRD*(0.6) * intensidade) + specularColor*0.2 ;

}


// nothing to do in here
extern "C" __global__ void __anyhit__radiance() {
}


// miss sets the bacgground color
extern "C" __global__ void __miss__radiance() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.7f, 1.0f);
}


// ray gen program - responsible for launching primary rays
extern "C" __global__ void __raygen__renderFrame() {

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  
    
    // ray payload
    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    const float2 screen(make_float2(ix+.5f,iy+.5f)
                    / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
  
    // note: nau already takes into account the field of view when computing 
    // camera horizontal and vertival
    float3 rayDir = normalize(camera.direction
                           + screen.x  * camera.horizontal
                           + screen.y * camera.vertical);
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
             camera.position,
             rayDir,
             0.f,    // tmin
             1e20f,  // tmax
             0.0f,   // rayTime
             OptixVisibilityMask( 255 ),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             PHONG_RAY_TYPE,             // SBT offset
             RAY_TYPE_COUNT,             // SBT stride
             PHONG_RAY_TYPE,             // missSBTIndex 
             u0, u1 );

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*pixelColorPRD.x);
    const int g = int(255.0f*pixelColorPRD.y);
    const int b = int(255.0f*pixelColorPRD.z);

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix+iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}


// nothing to do in here
extern "C" __global__ void __anyhit__shadow() {
    /*
    float3 &prd = *(float3*)getPRD<float3>();
    // set the shadow color
    prd = make_float3(0.0f, 0.0f, 0.0f);
    */
}
  

// nothing to do in here
extern "C" __global__ void __closesthit__shadow() {
    float3 &prd = *(float3*)getPRD<float3>();
    // set the shadow color
    prd = make_float3(0.0f, 0.0f, 0.0f);
}


// miss sets the bacgground color
extern "C" __global__ void __miss__shadow() {
    
    float3 &prd = *(float3*)getPRD<float3>();
    // set the background color
    prd = make_float3(1.0f, 1.0f, 1.0f);
    
}










extern "C" __global__ void __closesthit__phong_alphaTrans()
{

    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    float4 LightDir =optixLaunchParams.global->lightDir;
    
    // get the payload variable
    float3 &prd = *(float3*)getPRD<float3>();

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // get mesh data
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    
    float4 cor;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        cor = fromTexture;
    
    }
    else
        cor = make_float4(sbtData.color,1);

    //Aqui pegar a normal do pixel
    float4 normal = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    float4 pos
        = (1.f-u-v) * sbtData.vertexD.position[index.x]
        +         u * sbtData.vertexD.position[index.y]
        +         v * sbtData.vertexD.position[index.z];


   
    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                optixGetWorldRayDirection(),
                0.001f, // tmin
                1e20f, // tmax
                0.0f, // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                PHONG_RAY_TYPE, // SBT offset
                RAY_TYPE_COUNT, // SBT stride
                PHONG_RAY_TYPE, // missSBTIndex
                u0, u1 );   


    // I=L•N*C*I_l
    float intensidade = max(dot( -normalize(LightDir),normalize(normal)),0.f) ;


    prd =  make_float3(cor)*cor.w + pixelColorPRD*(1-cor.w);
   

}


// nothing to do in here
extern "C" __global__ void __anyhit__phong_alphaTrans() {
}


// miss sets the bacgground color
extern "C" __global__ void __miss__phong_alphaTrans() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(1.0f, 0.0f, 0.0f);
}


// nothing to do in here
extern "C" __global__ void __anyhit__shadow_alphaTrans() {
    /*
    float3 &prd = *(float3*)getPRD<float3>();
    // set the shadow color
    prd = make_float3(0.0f, 0.0f, 0.0f);
    */
}
  

// nothing to do in here
extern "C" __global__ void __closesthit__shadow_alphaTrans() {

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // get mesh data
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    float4 cor;

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        cor = fromTexture;    
    }

    float3 &prd = *(float3*)getPRD<float3>();
    // set the shadow color

    //lançar outro raio sombra para zonas com opacidade


    float4 pos
        = (1.f-u-v) * sbtData.vertexD.position[index.x]
        +         u * sbtData.vertexD.position[index.y]
        +         v * sbtData.vertexD.position[index.z];


    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                optixGetWorldRayDirection(),
                0.001f, // tmin
                1e20f, // tmax
                0.0f, // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                SHADOW, // SBT offset
                RAY_TYPE_COUNT, // SBT stride
                SHADOW, // missSBTIndex
                u0, u1 );   

    prd = pixelColorPRD*make_float3(1-cor.w,1-cor.w,1-cor.w);

}


// miss sets the bacgground color
extern "C" __global__ void __miss__shadow_alphaTrans() {
    
    float3 &prd = *(float3*)getPRD<float3>();
    // set the background color
    prd = make_float3(1.0f, 0.0f, 0.0f);
    
}






























extern "C" __global__ void __closesthit__phong_glass()
{

    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    float4 LightDir = optixLaunchParams.global->lightDir;
    
    // get the payload variable
    float3 &prd = *(float3*)getPRD<float3>();

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // get mesh data
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        prd= make_float3(fromTexture);
    
    }
    else
        prd = sbtData.color;

    //Aqui pegar a normal do pixel
    float4 normal = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];;
    //Calcular
    float4 light = normalize(LightDir);

    float value = max(dot(normalize(normal), light), 0.0);

    prd = prd * (1 - value);

    float3 rayDir = optixGetWorldRayDirection();
    float3 reflexDir = rayDir-2*dot(rayDir,make_float3(normal))*make_float3(normal);

    float4 pos
        = (1.f-u-v) * sbtData.vertexD.position[index.x]
        +         u * sbtData.vertexD.position[index.y]
        +         v * sbtData.vertexD.position[index.z];

    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                reflexDir,
                0.001f, // tmin
                1e20f, // tmax
                0.0f, // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                PHONG_RAY_TYPE, // SBT offset
                RAY_TYPE_COUNT, // SBT stride
                PHONG_RAY_TYPE, // missSBTIndex
                u0, u1 );   


    float3 pixelColorPRDTransparent = make_float3(1.f);
    uint32_t t0, t1;
    packPointer( &pixelColorPRDTransparent, t0, t1 );  

    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                rayDir,
                0.001f, // tmin
                1e20f, // tmax
                0.0f, // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                PHONG_RAY_TYPE, // SBT offset
                RAY_TYPE_COUNT, // SBT stride
                PHONG_RAY_TYPE, // missSBTIndex
                t0, t1 );   


    // I=L•N*C*I_l
    float intensidade = max(dot( -normalize(LightDir),normalize(normal)),0.f) ;

    prd = pixelColorPRDTransparent*0.8 + pixelColorPRD*0.1 + prd*0.1;

}


// nothing to do in here
extern "C" __global__ void __anyhit__phong_glass() {
}


// miss sets the bacgground color
extern "C" __global__ void __miss__phong_glass() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.7f, 1.0f);
}


// nothing to do in here
extern "C" __global__ void __anyhit__shadow_glass() {

}
  

// nothing to do in here
extern "C" __global__ void __closesthit__shadow_glass() {
    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    
    // get the payload variable
    float3 &prd = *(float3*)getPRD<float3>();

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // get mesh data
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;



    float4 pos
        = (1.f-u-v) * sbtData.vertexD.position[index.x]
        +         u * sbtData.vertexD.position[index.y]
        +         v * sbtData.vertexD.position[index.z];

 

    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                optixGetWorldRayDirection(),
                0.001f, // tmin
                1e20f, // tmax
                0.0f, // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                SHADOW, // SBT offset
                RAY_TYPE_COUNT, // SBT stride
                SHADOW, // missSBTIndex
                u0, u1 );   


    prd = pixelColorPRD;

}


// miss sets the background color
extern "C" __global__ void __miss__shadow_glass() {
    
    float3 &prd = *(float3*)getPRD<float3>();
    // set the background color
    prd = make_float3(1.0f, 1.0f, 1.0f);
    
}