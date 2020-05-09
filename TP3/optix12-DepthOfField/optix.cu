#include <optix.h>
#include "random.h"
#include "LaunchParams7.h" // our launch params
#include <vec_math.h> // NVIDIAs math utils

extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

//  a single ray type
enum { PHONG=0, SHADOW, RAY_TYPE_COUNT };

struct colorPRD{
    float3 color;
    unsigned int seed;
} ;

struct shadowPRD{
    float shadowAtt;
    unsigned int seed;
} ;


// -------------------------------------------------------
// closest hit computes color based lolely on the triangle normal

extern "C" __global__ void __closesthit__radiance()
{

    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    float4 lightPos = optixLaunchParams.global->lightPos;
    
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

    float4 pos
        = (1.f-u-v) * sbtData.vertexD.position[index.x]
        +         u * sbtData.vertexD.position[index.y]
        +         v * sbtData.vertexD.position[index.z];

    float4 lightDir = lightPos - pos;

    // I=L•N*C*I_l
    float intensidade = max(dot(-normalize(lightDir),normalize(normal)),0.f) ;

    optixTrace(optixLaunchParams.traversable,
                make_float3(pos),
                make_float3(-lightDir),
                0.001f, // tmins
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
    float3 lightReflect = make_float3(normalize(reflect(-lightDir,normal)));
    float specularFactor = dot(vertexToEye,lightReflect);
    if (specularFactor > 0) {
        int shininess = 1;
        specularFactor = pow(specularFactor,shininess);
        specularColor = pixelColorPRD * 1 * specularFactor;
    }

    // Componente ambiente + componente difusa*intensidade
    prd = (prd*0.2 + prd*pixelColorPRD*(0.8) * intensidade);

}

// nothing to do in here
extern "C" __global__ void __anyhit__radiance() {
}


// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.7f, 1.0f);
}


// -----------------------------------------------
// Shadow rays

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

// miss sets the background color
extern "C" __global__ void __miss__shadow() {
    
    float3 &prd = *(float3*)getPRD<float3>();
    // set the background color
    prd = make_float3(1.0f, 1.0f, 1.0f);
    
}




// -----------------------------------------------
// Primary Rays

extern "C" __global__ void __raygen__renderFrame() {

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  
    
	if (optixLaunchParams.frame.frame == 0 && ix == 0 && iy == 0) {
		// print info to console
		printf("===========================================\n");
        printf("Nau Ray-Tracing Debug\n");
        const float4 &ld = optixLaunchParams.global->lightPos;
        printf("LightPos: %f, %f %f %f\n", ld.x,ld.y,ld.z,ld.w);
        printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
        printf("Rays per pixel squared: %d \n", optixLaunchParams.frame.raysPerPixel);
		printf("===========================================\n");
	}

    float lensDistance  = optixLaunchParams.global->lensDistance;
    float focalDistance = optixLaunchParams.global->focalDistance;
    float aperture = optixLaunchParams.global->aperture;
    float3 frente = normalize(cross(camera.vertical,camera.horizontal));
    float3 lensCentre = camera.position + frente*lensDistance;

    // ray payload
    colorPRD pixelColorPRD;
    pixelColorPRD.color = make_float3(1.f);

    float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    // half pixel
    float2 delta = make_float2(1.0f/raysPerPixel, 1.0f/raysPerPixel);

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
  
    float red = 0.0f, blue = 0.0f, green = 0.0f;
    for (int i = 0; i < raysPerPixel; ++i) {
        for (int j = 0; j < raysPerPixel; ++j) {

            uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, i*raysPerPixel + j );

            pixelColorPRD.seed = seed;
            uint32_t u0, u1;
            packPointer( &pixelColorPRD, u0, u1 );  
            //const float2 subpixel_jitter = make_float2( i * delta.x + delta.x *  rnd( seed ), j * delta.y + delta.y * rnd( seed ) );
            //const float2 subpixel_jitter = make_float2( rnd( seed )-0.5f, rnd( seed )-0.5f );
            const float2 subpixel_jitter = make_float2(i * delta.x, j * delta.y);
            const float2 screen(make_float2(ix + subpixel_jitter.x, iy + subpixel_jitter.y)
                            / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
        
        // note: nau already takes into account the field of view and ratio when computing 
        // camera horizontal and vertival

            float3 cPos = camera.position+(-screen.x)*camera.horizontal + (-screen.y ) * camera.vertical;

            float3 rayDir = normalize(lensCentre - cPos);

            //float3 proj_frente_rayDir = (dot(rayDir,frente)/(length(frente)*length(frente)))*frente; //frente ta normalizado POSSO SIMPLIFICAR
            float3 proj_frente_rayDir = dot(rayDir,frente)*frente;


            // Vetor que vai do centro da lente para o ponto de foco no plano de foco
            float3 ray = rayDir * focalDistance / length(proj_frente_rayDir);

            float3 pFocal = lensCentre + ray;


            float randR = aperture * sqrt(rnd(seed));

            float randA = rnd(seed) * 2 * M_PIf;

            float x = randR * cos(randA);
            float y = randR * sin(randA);

            float3 randAperture = lensCentre + camera.horizontal * x + camera.vertical * y;//make_float3(((lensCentre.x + camera.horizontal * x),(lensCentre.y + camera.vertical * y),lensCentre.z));

            float3 rayDirection = pFocal - randAperture;
            
            // trace primary ray
            optixTrace(optixLaunchParams.traversable,
                    randAperture,
                    rayDirection,
                    0.f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_NONE,//,OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    PHONG,             // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    PHONG,             // missSBTIndex 
                    u0, u1 );

            red += pixelColorPRD.color.x / (raysPerPixel*raysPerPixel);
            green += pixelColorPRD.color.y / (raysPerPixel*raysPerPixel);
            blue += pixelColorPRD.color.z / (raysPerPixel*raysPerPixel);
        }
    }

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*red);
    const int g = int(255.0f*green);
    const int b = int(255.0f*blue);
    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
