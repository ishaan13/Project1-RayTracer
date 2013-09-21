-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2013 - Ishaan Singh
-------------------------------------------------------------------------------
Due Thursday, 09/19/2013
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
Features:
-------------------------------------------------------------------------------
* Ray tracing per bounce. Compaction not completed yet.
* Specular Reflection (make has reflection between 0.1 and 1)
* Soft Shadows and Area Lights
* Depth of Field
* Jittered Antialiasing
* MSAA: Multi Sample Anti Aliasing
* Refraction (make has refraction between 0.1 and 1)
* Camera Movement
  w-a-s-d-q-z move camera in world space
  [-]-o-p pan camera around
  up - down for moving the focal plan in and out
  left - right for increasing and decreasing aperture size
  
-------------------------------------------------------------------------------
Images:
-------------------------------------------------------------------------------
http://github.com/ishaan13/Project1-RayTracer/blob/master/renders/2K_10_noCompaction.bmp
http://github.com/ishaan13/Project1-RayTracer/blob/master/renders/2K_10_noCompaction_jitter.bmp
http://github.com/ishaan13/Project1-RayTracer/blob/master/renders/2K_10_noCompaction_jitter_DOF.bmp

-------------------------------------------------------------------------------
Videos:
-------------------------------------------------------------------------------
https://www.dropbox.com/s/htm8t0xxahi0z1l/RayTracerGPU.avi
https://www.dropbox.com/s/xnmadcdf24i5zr4/RayTracerGPUDOF.avi

-------------------------------------------------------------------------------
Performance Evaluation:
-------------------------------------------------------------------------------
Peformance Boosts
* Ray-Box intersection code has less branching
* Bounce Parallelization

Performance increase strategy:
* Compaction should increase speed massively because of  decrease in dead threads
* Put light source geometry into shared memory
* Less global memory access in code. Use more register-variables

Performance Analysis:

Varying Number of threads per block and looking at time per iteration for a maximum of 10 bounces:
on NVidia GT640M (384 cores, 1GB)

*  4 x  4 : 395ms
*  8 x  8 : 120ms
* 12 x 12 : 139ms
* 16 x 16 : 125ms
* 24 x 24 : 229ms
* 32 x 32 : 156ms

So, the interesting thing is that at multiples of a half-warp size per dimension, there's a 
significant increase in performance as all the local maxima's. It would be curious to understand why.
The reason being that even the 24 x 24 is a multiple of a 32 warp size.
