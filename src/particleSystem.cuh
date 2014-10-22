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

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void copyArrayToHost(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(SimParams *hostParams);
    //void setCellParameters(CellParams *hostParams);

    void integrateSystem(float *pos,
                         float *vel,
                         float deltaTime,
                         uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void calcNewHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void calcNewVelHashMax(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    maxNumParticles);

    void calcNewHashMax(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    maxNumParticles);

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
                                     uint   numCells);

/*    void boundaryFlux(
        //m_dVel,               // receive new velocities here
        float *sortedPos,         // send off the sorted positions
        //m_dSortedVel,         // send off the sorted velocities
        uint *gridParticleIndex,
        uint *cellStart,
        uint *cellEnd,
        uint& numParticles,
        uint numCells);
*/
/*    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);
*/
    void advect (float *newVel,
                 float *sortedPos,
              //   float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells,
                 float *randomArray,
                 float *Uz,
                 float *Ur,
                 float *EpsTKE,
                 float *Nut,
                 float *GradZNut,
                 float *GradRNut,
                 float *DUz,
                 float *DUr
     			);

    void chemistry(
    		float *sortedPos,
    		float *sortedScalar,
    		float *sortedDiamBins,
    		float *colors,
    		uint maxNumParticles,
    		uint numParticles
    		);

    void balanceCellsCuda(            // calculate and return velocities
    			// float *newPos,
    			// float *newScalar,
                 float *sortedPos,         // accept the positions (sorted)
    			 float *sortedScalar,
    			 float *sortedBinDiams,
                 uint  *gridParticleHash,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   maxNumParticles,
                 uint  &numParticles,
                 uint   numCells);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
