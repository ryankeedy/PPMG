/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void cudaGLInit(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void copyArrayToHost(void *host, const void *device, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) host + offset, device, size, cudaMemcpyDeviceToHost));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));

        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

/*    void setCellParameters(CellParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(cellparams, hostParams, sizeof(CellParams)));
    }*/

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateSystem(float *pos,
                         float *vel,
                         //float *scalar,
                         float deltaTime,
                         uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);
        //thrust::device_ptr<float2>  d_scalar((float2 *)scalar);

        thrust::for_each(
        	thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            //thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_scalar)),
        	thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles)),
            //thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles, d_scalar+numParticles)),
            integrate_functor(deltaTime));
    }

    void calcNewHash(uint  *gridParticleHash,   // will be determined
                  uint  *gridParticleIndex,  // will be determined
                  float *pos,                // input that will dictate hash
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        printf("Diving into calcHashD kernel\n");
        printf("  with %d blocks and %d threads\n",numBlocks,numThreads);
        // execute the kernel
        calcNewHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: calcHash");

    	threadSync();

    }

    void calcNewVelHashMax(uint  *gridParticleHash,   // will be determined
                  uint  *gridParticleIndex,  // will be determined
                  float *pos,                // input that will dictate hash
                  int    maxNumParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(maxNumParticles, 256, numBlocks, numThreads);

        printf("Diving into calcNewVelHashMaxD kernel\n");
        printf("  with %d blocks and %d threads\n",numBlocks,numThreads);
        // execute the kernel
        calcNewVelHashMaxD<<< numBlocks, numThreads >>>(gridParticleHash,
        											 gridParticleIndex,
                                                     (float4 *) pos,
                                                     maxNumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: calcVelHash");

    	threadSync();

    }

    void calcNewHashMax(uint  *gridParticleHash,   // will be determined
                  uint  *gridParticleIndex,  // will be determined
                  float *pos,                // input that will dictate hash
                  int    maxNumParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(maxNumParticles, 256, numBlocks, numThreads);

        printf("Diving into calcNewHashMaxD kernel\n");
        printf("  with %d blocks and %d threads\n",numBlocks,numThreads);
        // execute the kernel
        calcNewHashMaxD<<< numBlocks, numThreads >>>(gridParticleHash,
        											 gridParticleIndex,
                                                     (float4 *) pos,
                                                     maxNumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: calcHash");

    	threadSync();

    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     float *sortedScalar,
                                     float *sortedDiamBins,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     float *oldScalar,
                                     float *oldDiamBins,
                                     uint   maxNumParticles,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
//        computeGridSize(maxNumParticles, 256, numBlocks, numThreads);
        computeGridSize(maxNumParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        printf("Setting cells %d\n",numCells);
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
        checkCudaErrors(cudaMemset(cellEnd, 0xffffffff, numCells*sizeof(uint)));
        printf("cells Set\n");
        //checkCudaErrors(cudaMemset(cellStart, (uint)99, numCells*sizeof(uint)));
        //checkCudaErrors(cudaMemset(cellEnd, (uint)(numCells+1), numCells*sizeof(uint)));

//#if USE_TEX
//        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, maxNumParticles*sizeof(float4)));
//        checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, maxNumParticles*sizeof(float4)));
//#endif

        uint smemSize = sizeof(uint)*(numThreads+1);
        printf("before reorder\n");
        reorderDataAndFindCellStartMaxD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (float4 *) sortedPos,
            (float4 *) sortedVel,
            sortedScalar,
            sortedDiamBins,
            gridParticleHash,
            gridParticleIndex,
            (float4 *) oldPos,
            (float4 *) oldVel,
            oldScalar,
            oldDiamBins,
            maxNumParticles,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartMaxD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
        threadSync();

        printf("After reorder\n");

        //printf("inspect again %d\n",cellEnd[0]);
        //printf("inspect again %f\n",sortedScalar[0]);

    }

    void advect (float *newVel,            // calculate and return velocities
                 float *sortedPos,         // accept the positions (sorted)
               //  float *sortedVel,         // accept the velocities (sorted) [NOT NEEDED]
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 //uint   numParticles,
                 uint   maxNumParticles,
                 uint   numCells,
                 float *randomArray, //)
                 //float *random2)
     			 float *Uz,
     			 float *Ur,
     			 float *epsTKE,
     			 float *nut,
     			 float *gradZNut,
     			 float *gradRNut,
     			 float *dUz,
     			 float *dUr)

    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, maxNumParticles*sizeof(float4)));
     //   checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, maxNumParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(maxNumParticles, 64, numBlocks, numThreads);

        // execute the kernel
        advectD<<< numBlocks, numThreads >>>((float4 *)newVel,
                                              (float4 *)sortedPos,
                                             // (float4 *)sortedVel,
                                              gridParticleIndex,
                                              cellStart,
                                              cellEnd,
                                              maxNumParticles,
                                              randomArray,
                                  			  Uz,
                                  			  Ur,
                                  			  //epsTKE,
                                  			  nut,
                                  			  gradZNut,
                                  			  gradRNut,
                                  			  dUz,
                                  			  dUr);
                                              //random2);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
    }

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint maxNumParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + maxNumParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
        threadSync();
    }

    void mixCurlCuda(float *newScalar,
                 float *sortedScalar,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   maxNumParticles,
                 uint   numCells)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(maxNumParticles, 64, numBlocks, numThreads);

        // execute the kernel
        mixCurlD<<< numBlocks, numThreads >>>(newScalar,
                                              sortedScalar,
                                              gridParticleIndex,
                                              cellStart,
                                              cellEnd,
                                              maxNumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: mixCurl");
    }

    void balanceCellsCuda(            // calculate and return velocities
    			 //float *newPos,
    			 //float *newScalar,
                 float *sortedPos,         // accept the positions (sorted)
    			 float *sortedScalar,
    			 float *sortedDiamBins,
                 uint  *gridParticleHash,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   maxNumParticles,
                 uint   &numParticles,
                 uint   numCells)
    {

        // thread per cell
        uint numThreads, numBlocks;
        computeGridSize(numCells, 64, numBlocks, numThreads);
        printf("nb, nt %d %d %d %d\n",numBlocks,numThreads,numCells,numParticles);

        //uint smemSize = sizeof(uint)*(numThreads+1);
        int hLessParts=0;
        int hNumParts=(int)numParticles;
        //int hNumParts=numParticles;

        int* dLessParts;
        int* dNumParts;
        //int dLessParts;
        //int dNumParts;

        ////checkCudaErrors(cudaMalloc((void**)&dLessParts, sizeof(uint)));
        checkCudaErrors(cudaMalloc((void **)  &dLessParts,sizeof(int)));
        checkCudaErrors(cudaMemcpy(dLessParts,&hLessParts,sizeof(int),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)  &dNumParts, sizeof(int)));
        checkCudaErrors(cudaMemcpy(dNumParts, &hNumParts, sizeof(int),cudaMemcpyHostToDevice));
        ////copyArrayToDevice(dNumParts,hNumParts, 0, sizeof(uint));
        //*lessParticles = 0;
        printf("balanceD %d %d %d %d\n",hNumParts,hLessParts,numParticles,hNumParts);
        //cin.ignore();
        // execute the kernel
        //mixCurlD<<< numBlocks, numThreads >>>(newScalar,
/**/       balanceCellsD<<< numBlocks, numThreads >>>(  //, smemSize
        	//	(float4 *)newPos,
       		// newScalar,
       		(float4 *)sortedPos,         // accept the positions (sorted)
       		 sortedScalar,
       		 sortedDiamBins,
             gridParticleHash,
             gridParticleIndex,
             cellStart,
             cellEnd,
              dNumParts,
              dLessParts);
/**/
        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: balanceCells");

        //threadSync;
        checkCudaErrors(cudaMemcpy(&hLessParts,dLessParts,sizeof(int),cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&hNumParts, dNumParts, sizeof(int),cudaMemcpyDeviceToHost));

        getLastCudaError("Kernel execution failed: balanceCells back to host");

        printf("balanceD after %d %d\n",hNumParts,hLessParts);
        numParticles = hNumParts - hLessParts;

    }

    void chemistry(            // calculate and return velocities
                 float *sortedPos,         // accept the positions (sorted)
    			 float *sortedScalar,
    			 float *sortedDiamBins,
    			 float *colors,    // output colors
                 //uint  *gridParticleHash,
                 //uint  *gridParticleIndex,
                 //uint  *cellStart,
                 //uint  *cellEnd,
                 uint   maxNumParticles,
                 uint   numParticles)
    {

        // thread per cell
        uint numThreads, numBlocks;
        computeGridSize(maxNumParticles, 64, numBlocks, numThreads);
        //printf("nb, nt %d %d\n",numBlocks,numThreads);

        // execute the kernel
/**/       chemistryD<<< numBlocks, numThreads >>>(  //, smemSize
        	//	(float4 *)newPos,
       		// newScalar,
       		(float4 *)sortedPos,         // accept the positions (sorted)
       		 sortedScalar,
       		 sortedDiamBins,
       		 colors,
            // gridParticleHash,
            // gridParticleIndex,
            // cellStart,
            // cellEnd,
             // maxNumParticles,
              numParticles);
/**/
        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: chemistry");

    }

}   // extern "C"
