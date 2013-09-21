// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

// Optimizations and add ons
#define JITTER 1
#define COMPACTION 0
#define ACCUMULATION 1
#define DOF 1

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* accumulatorImage = NULL;
extern bool singleFrameMode;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	getchar();
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov
#if DOF
											,float aperture, float focusPlane
#endif
	){
  ray r;

  // @DO: verify field of view!
  glm::vec3 axis_a = glm::cross(view, up);
  glm::vec3 axis_b = glm::cross(axis_a, view);
  glm::vec3 midPoint = eye + view;
  glm::vec3 viewPlaneX = axis_a * tan(fov.x) * glm::length(view)/glm::length(axis_a);
  glm::vec3 viewPlaneY = axis_b * tan(fov.y) * glm::length(view)/glm::length(axis_b);

#if JITTER
  glm::vec3 jitter = generateRandomNumberFromThread(resolution,time,x,y);
  glm::vec3 screenPoint = midPoint +
							(2.0f * ((jitter.x + 1.0f * x) / (resolution.x-1)) - 1.0f) * viewPlaneX + 
							(2.0f * ((jitter.y + 1.0f * y) / (resolution.y-1)) - 1.0f) * viewPlaneY;
#else
  glm::vec3 screenPoint = midPoint +
							(2.0f * (1.0f * x / (resolution.x-1)) - 1.0f) * viewPlaneX + 
							(2.0f * (1.0f * y / (resolution.y-1)) - 1.0f) * viewPlaneY;

#endif

#if DOF

  glm::vec3 focusPlaneIntersection;
  float focalPlaneDepth;
  
  r.origin = eye;
  r.direction = glm::normalize(screenPoint - eye);

  glm::vec3 focusPlaneCenter = r.origin + r.direction * focusPlane;
  planeIntersectionTest(focusPlaneCenter,view,r,focusPlaneIntersection);

  glm::vec3 apertureJitter = aperture * (generateRandomNumberFromThread(resolution,time,x,y) - 0.5f);
  r.origin = r.origin + apertureJitter;
  r.direction = glm::normalize(focusPlaneIntersection - r.origin);

#else
  r.origin = screenPoint;
  r.direction = glm::normalize(screenPoint - eye);
#endif
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__device__ bool visibilityCheck(ray r, staticGeom* geoms, int numberOfGeoms, glm::vec3 pointToCheck, int geomShotFrom)
{
	bool visible = true;
	float distance = glm::length(r.origin - pointToCheck);

	// Check whether any object occludes point to check from ray's origin
	for(int iter=0; iter < numberOfGeoms; iter++)
	{
		float depth=-1;
		glm::vec3 intersection;
		glm::vec3 normal;
		
		if(geoms[iter].type == CUBE)
		{
			depth = boxIntersectionTest(geoms[iter],r,intersection,normal);
		}
		
		
		else if(geoms[iter].type == SPHERE)
		{
			depth = sphereIntersectionTest(geoms[iter],r,intersection,normal);
		}
		
		if(depth > 0 && depth < distance)
		{
			//printf("Depth: %f\n", depth);
			visible = false;
			break;
		}
	}

	
	return visible;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, ray* rayPool, float *transmission){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y) && transmission[index] > FLOAT_EPSILON){

	ray r = rayPool[index];	


	//Check all geometry for intersection
	int intersectedGeom = -1;
	float minDepth = 1000000.0f;
	glm::vec3 minIntersectionPoint;
	glm::vec3 minNormal = glm::vec3(0.0f);
	for(int iter=0; iter < numberOfGeoms; iter++)
	{
		float depth=-1;
		glm::vec3 intersection;
		glm::vec3 normal;
		if(geoms[iter].type == CUBE)
		{
			depth = boxIntersectionTest(geoms[iter],r,intersection,normal);
		}
		
		else if(geoms[iter].type == SPHERE)
		{
			depth = sphereIntersectionTest(geoms[iter],r,intersection,normal);
		}
		

		if(depth > 0 && depth < minDepth)
		{
			minDepth = depth;
			minIntersectionPoint = intersection;
			minNormal = normal;
			intersectedGeom = iter;
		}
	}

	// Depth render - test
	//float maxDepth = 15.0f;
	
	glm::vec3 diffuseLight = glm::vec3(0.0f);
	glm::vec3 phongLight = glm::vec3(0.0f);

	glm::vec3 diffuseColor;
	glm::vec3 specularColor;
	glm::vec3 emittance;

	//Calculate Lighting if any geometry is intersected
	if(intersectedGeom > -1)
	{
		//finalColor = materials[geoms[intersectedGeom].materialid].color;
		material m = materials[geoms[intersectedGeom].materialid];
		diffuseColor = m.color;
		specularColor = m.specularColor;
		emittance = m.color * m.emittance;

		// Stochastic Diffused Lighting with "area" lights
		for(int iter = 0; iter < numberOfGeoms; iter++)
		{
			material lightMaterial = materials[geoms[iter].materialid];
			// If this geometry is going to act like a light source
			if(lightMaterial.emittance > 0.0001f)
			{
				glm::vec3 lightSourceSample;

				// Get a random point on the light source
				if(geoms[iter].type == SPHERE)
				{
					lightSourceSample = getRandomPointOnSphere(geoms[iter],time*index);
				}
				else if(geoms[iter].type == CUBE)
				{
					lightSourceSample = getRandomPointOnCube(geoms[iter],time*index);
				}

				// Diffuse Lighting Calculation
				glm::vec3 L = glm::normalize(lightSourceSample - minIntersectionPoint);
				
				//Shadow Ray check
				ray shadowRay;
				shadowRay.origin = minIntersectionPoint + NUDGE * L;
				shadowRay.direction = L;

				bool visible = visibilityCheck(shadowRay,geoms,numberOfGeoms,lightSourceSample, intersectedGeom);

				if(visible)
				{
					diffuseLight += lightMaterial.color * lightMaterial.emittance * glm::max(glm::dot(L,minNormal),0.0f);

					// Calculate Phong Specular Part only if exponent is greater than 0
					if(m.specularExponent > FLOAT_EPSILON)
					{
						glm::vec3 reflectedLight = 2.0f * minNormal * glm::dot(minNormal, L) - L;
						phongLight += lightMaterial.color * lightMaterial.emittance * pow(glm::max(glm::dot(reflectedLight,minNormal),0.0f),m.specularExponent);
					}

				}
			}
		}

		AbsorptionAndScatteringProperties absScatProps;
		glm::vec3 colorSend, unabsorbedColor;
		int rayPropogation = calculateBSDF(r,minIntersectionPoint,minNormal,diffuseColor*m.emittance,absScatProps,colorSend,unabsorbedColor,m);
		
		// Reflection; calculate transmission coeffiecient
		if(rayPropogation == 1)
		{
			colors[index] += transmission[index] * (1.0f - m.hasReflective) * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
			transmission[index] *= m.hasReflective;
			rayPool[index] = r;
		}
		// Refraction; calculate transmission coeffiecient
		else if (rayPropogation == 2)
		{
			colors[index] += transmission[index] * (1.0f - m.hasRefractive) * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
			transmission[index] *= m.hasRefractive;
			rayPool[index] = r;
		}
		// Diffuse Surface or Light, mark ray as dead
		else
		{
			colors[index] += transmission[index] * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
			transmission[index] = 0;
		}

	}
	// No intersection, mark rays as dead
	// Ambeint term 
	else
	{
		glm::vec3 ambient = glm::vec3(0,0,0);
		colors[index] += ambient; 
		transmission[index] = 0;
	}
	
	/*
		//Checking for correct ray direction
		colors[index].x = fabs(r.direction.x);
		colors[index].y = fabs(r.direction.y);
		colors[index].z = fabs(r.direction.z);
	
		//Check for correct material pickup
		colors[index] = color;

		//Checking for correct depth testing
		colors[index] = color * (maxDepth - minDepth)/maxDepth;

		//Checking for correct normals
		colors[index] = glm::vec3(minNormal);
		colors[index] = glm::vec3( fabs(minNormal.x), fabs(minNormal.y), fabs(minNormal.z));
	*/

	
	
   }
}

__global__ void fillRayPoolFromCamera(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, ray* rayPool, float *transmission){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){

	rayPool[index] = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov
#if DOF
											,cam.aperture, cam.focusPlane
#endif
		);
	transmission[index] = 1.0f;
   }
}

__global__ void combineIntoAccumulatorImage(glm::vec2 resolution, float frames, glm::vec3* inputColors, glm::vec3* displayColors)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
	  displayColors[index] = (((frames-1) * displayColors[index]) + inputColors[index])/frames;
  }
}

//__global__ void calculateIndices(ray* inputRays, ray* outputRays, 

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, cameraData liveCamera){


  if(iterations == 1)
  {
    // Allocate Accumulator Image
    cudaAllocateAccumulatorImage(renderCam);
  }

  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //create buffer for holding indices to write to

  //create buffer for holding transmission coefficeint
  float* cudaTransmissionCoeffecient;
  cudaMalloc((void**)&cudaTransmissionCoeffecient, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(float));

  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //user interaction
  cam.position +=  (liveCamera.position);
  cam.view = glm::normalize(cam.view + liveCamera.view);
  cam.aperture = liveCamera.aperture;
  cam.focusPlane = liveCamera.focusPlane;

  //Transfer materials
  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //Create Memory for RayPool
  ray* cudarays = NULL;
  cudaMalloc((void**)&cudarays, (renderCam->resolution.x * renderCam->resolution.y) * sizeof(ray));

  //clear On screen buffer
  clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution, cudaimage);

  //Fill ray pool with rays from camera for first iteration
  fillRayPoolFromCamera<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudarays, cudaTransmissionCoeffecient);

  for(int i=0; i < MAX_RECURSION_DEPTH; i++)
  {
	//kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, cudarays, cudaTransmissionCoeffecient);
  }

#if ACCUMULATION
  combineIntoAccumulatorImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudaimage, accumulatorImage);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, accumulatorImage);
#else
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
#endif

  //retrieve image from GPU for sending to bmp file
  if(singleFrameMode)
#if ACCUMULATION
	cudaMemcpy( renderCam->image, accumulatorImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree( cudarays );
  cudaFree( cudaTransmissionCoeffecient );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

//Clear AccumulatorImage. For an interactive application, this needs to be called everytime the camera moves or the scene changes
void cudaClearAccumulatorImage(camera *renderCam)
{
	// set up crucial magic
  	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution, accumulatorImage);
}

//Allocate Memory For AccumulatorImage
void cudaAllocateAccumulatorImage(camera *renderCam)
{
	cudaMalloc((void**)&accumulatorImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaClearAccumulatorImage(renderCam);
}

//Free memory of the accumulator image
void cudaFreeAccumulatorImage()
{
	cudaFree(accumulatorImage);
}