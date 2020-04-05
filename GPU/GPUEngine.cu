/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
//#include "../hash/sha256.h"
//#include "../hash/ripemd160.h"
#include "../Timer.h"

//#include "GPUGroup.h"
#include "GPUSptable.h"

#include "GPUMath.h"
//#include "GPUHash.h"
//#include "GPUBase58.h"
//#include "GPUWildcard.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------
// uint64_t *pkeys  - Tame point
// uint64_t *pwkeys - Wild point
// skeys + xPtr - Tame Start keys
// skeys + yPtr - Wils Start keys
__global__ void comp_keys(uint64_t *pkeys, uint64_t *pwkeys, uint64_t *skeys, uint64_t DPmodule, uint32_t hop_modulo, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * NB_TRHEAD_PER_GROUP;
  
  ComputeKeys(pkeys + xPtr, pkeys + yPtr, pwkeys + xPtr, pwkeys + yPtr, skeys + xPtr, skeys + yPtr, DPmodule, hop_modulo, maxFound, found);

}

// ---------------------------------------------------------------------------------------

using namespace std;

/*
std::string toHex(unsigned char *data, int length) {

  string ret;
  char tmp[3];
  for (int i = 0; i < length; i++) {
    if (i && i % 4 == 0) ret.append(" ");
    sprintf(tmp, "%02x", (int)data[i]);
    ret.append(tmp);
  }
  return ret;

}
*/

int _ConvertSMVer2Cores(int major, int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1} };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

GPUEngine::GPUEngine(int nbThreadGroup, int gpuId, uint32_t maxFound, bool rekey, uint32_t pow2w, uint32_t hop_modulo, int power) {

  // Initialise CUDA
  this->rekey = rekey;
  initialised = false;
  cudaError_t err;
  
  //int pow2dp = (pow2w/2)-2;
  //int pow2dp = 22;// Fixed
  
  int pow2dp = ((pow2w-(2*power))/2)-2;  
  
  if (pow2dp > 24) {
	printf("GPUEngine: Old DPmodule = 2^%d \n", pow2dp);
	pow2dp = 24;
	printf("GPUEngine: New DPmodule = 2^%d \n", pow2dp);
  }
  if (pow2dp < 18) {
	printf("GPUEngine: Old DPmodule = 2^%d \n", pow2dp);
	pow2dp = 18;
	printf("GPUEngine: New DPmodule = 2^%d \n", pow2dp);
  }
  
  uint64_t DPmodule = (uint64_t)1 << pow2dp;
  
  //printf("GPUEngine: Fixed DPmodule: 0x%lx 2^%d Hop_modulo: %d Power: %d \n", DPmodule, pow2dp, hop_modulo, power);
  printf("GPUEngine: DPmodule: 0x%lx 2^%d ((pow2W-(2*Power))/2)-2 Hop_modulo: %d Power: %d \n", DPmodule, pow2dp, hop_modulo, power);
  
  this->DPmodule = DPmodule;
  this->hop_modulo = hop_modulo;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  if (nbThreadGroup == -1)
    nbThreadGroup = deviceProp.multiProcessorCount * 8;

  this->nbThread = nbThreadGroup * NB_TRHEAD_PER_GROUP;
  this->maxFound = maxFound;
  //this->outputSize = (maxFound*ITEM_SIZE + 4);
  this->outputSize = (maxFound*ITEM_SIZE);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
  gpuId,deviceProp.name,deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
  nbThread / NB_TRHEAD_PER_GROUP,
  NB_TRHEAD_PER_GROUP);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  size_t stackSize = 49152;
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  /*
  size_t heapSize = ;
  err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(0);
  }

  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  printf("Stack Size %lld\n", size);
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  printf("Heap Size %lld\n", size);
  */

  // Allocate memory

  err = cudaMalloc((void **)&inputKey, nbThread * 32 * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&w_inputKey, nbThread * 32 * 2);// add Wild point
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&s_inputKey, nbThread * 32 * 2);// add Tame and Wild Start keys
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&w_inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);// add Wild point
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&s_inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);// add Tame and Wild Start keys
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }  
  err = cudaMalloc((void **)&outputPrefix, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  initialised = true;

}

int GPUEngine::GetGroupSize() {
  return GRP_SIZE;
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
     NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i=0;i<deviceCount;i++) {

    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      deviceProp.major, deviceProp.minor,(double)deviceProp.totalGlobalMem/1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

GPUEngine::~GPUEngine() {

  cudaFree(inputKey);// Tame point
  cudaFree(w_inputKey);// Wild point
  cudaFree(s_inputKey);// Tame and Wild Start keys
  
  cudaFreeHost(outputPrefixPinned);
  cudaFree(outputPrefix);

}

int GPUEngine::GetNbThread() {
  return nbThread;
}



bool GPUEngine::callKernel() {

  // Reset nbFound
  //cudaMemset(outputPrefix,0,4); <-- copy duplicates to host
  cudaMemset(outputPrefix,0,outputSize);
  

  // Call the kernel 
  if (1) {
  
	// ;)
	comp_keys << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> >
		(inputKey, w_inputKey, s_inputKey, DPmodule, hop_modulo, maxFound, outputPrefix);
    
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;

}

bool GPUEngine::SetKeys(Point *p, Point *wp, Int *keys, Int *wkeys) {//bool GPUEngine::SetKeys(Point *p) {

  // Sets the starting keys for each thread
  // p must contains nbThread public keys
  for (int i = 0; i < nbThread; i+= NB_TRHEAD_PER_GROUP) {
    for (int j = 0; j < NB_TRHEAD_PER_GROUP; j++) {

      // Tame point
	  inputKeyPinned[8*i + j + 0*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[0];
      inputKeyPinned[8*i + j + 1*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[1];
      inputKeyPinned[8*i + j + 2*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[2];
      inputKeyPinned[8*i + j + 3*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[3];

      inputKeyPinned[8*i + j + 4*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[0];
      inputKeyPinned[8*i + j + 5*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[1];
      inputKeyPinned[8*i + j + 6*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[2];
      inputKeyPinned[8*i + j + 7*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[3];
	  
	  // add Wild point
	  w_inputKeyPinned[8*i + j + 0*NB_TRHEAD_PER_GROUP] = wp[i + j].x.bits64[0];
      w_inputKeyPinned[8*i + j + 1*NB_TRHEAD_PER_GROUP] = wp[i + j].x.bits64[1];
      w_inputKeyPinned[8*i + j + 2*NB_TRHEAD_PER_GROUP] = wp[i + j].x.bits64[2];
      w_inputKeyPinned[8*i + j + 3*NB_TRHEAD_PER_GROUP] = wp[i + j].x.bits64[3];
	  
	  w_inputKeyPinned[8*i + j + 4*NB_TRHEAD_PER_GROUP] = wp[i + j].y.bits64[0];
      w_inputKeyPinned[8*i + j + 5*NB_TRHEAD_PER_GROUP] = wp[i + j].y.bits64[1];
      w_inputKeyPinned[8*i + j + 6*NB_TRHEAD_PER_GROUP] = wp[i + j].y.bits64[2];
      w_inputKeyPinned[8*i + j + 7*NB_TRHEAD_PER_GROUP] = wp[i + j].y.bits64[3];
	  
	  // add Start keys Tame and Wild
	  s_inputKeyPinned[8*i + j + 0*NB_TRHEAD_PER_GROUP] = keys[i + j].bits64[0];// Tame Start keys
      s_inputKeyPinned[8*i + j + 1*NB_TRHEAD_PER_GROUP] = keys[i + j].bits64[1];
      s_inputKeyPinned[8*i + j + 2*NB_TRHEAD_PER_GROUP] = keys[i + j].bits64[2];
      s_inputKeyPinned[8*i + j + 3*NB_TRHEAD_PER_GROUP] = keys[i + j].bits64[3];

      s_inputKeyPinned[8*i + j + 4*NB_TRHEAD_PER_GROUP] = wkeys[i + j].bits64[0];// Wild Start keys
      s_inputKeyPinned[8*i + j + 5*NB_TRHEAD_PER_GROUP] = wkeys[i + j].bits64[1];
      s_inputKeyPinned[8*i + j + 6*NB_TRHEAD_PER_GROUP] = wkeys[i + j].bits64[2];
      s_inputKeyPinned[8*i + j + 7*NB_TRHEAD_PER_GROUP] = wkeys[i + j].bits64[3];

    }
  }
  
  // Fill device memory
  cudaMemcpy(inputKey, inputKeyPinned, nbThread*32*2, cudaMemcpyHostToDevice);
  cudaMemcpy(w_inputKey, w_inputKeyPinned, nbThread*32*2, cudaMemcpyHostToDevice);// Wild point
  cudaMemcpy(s_inputKey, s_inputKeyPinned, nbThread*32*2, cudaMemcpyHostToDevice);// Tame and Wild Start keys
  
  if (!rekey) {
    // We do not need the input pinned memory anymore
    cudaFreeHost(inputKeyPinned);
    inputKeyPinned = NULL;
	// add
	cudaFreeHost(w_inputKeyPinned);// add Wild point
	w_inputKeyPinned = NULL;
	// add
	cudaFreeHost(s_inputKeyPinned);// add Tame and Wild Start keys
	s_inputKeyPinned = NULL;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
  }

  return callKernel();

}

bool GPUEngine::Launch(std::vector<ITEM> &prefixFound, bool spinWait) {

  prefixFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputPrefixPinned[0];
  if (nbFound > maxFound) {
    // prefix has been lost
    if (!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n", (nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }
  
  // When can perform a standard copy, the kernel is eneded
  //cudaMemcpy( outputPrefixPinned , outputPrefix , nbFound*ITEM_SIZE + 4 , cudaMemcpyDeviceToHost);
  cudaMemcpy( outputPrefixPinned , outputPrefix , nbFound*ITEM_SIZE, cudaMemcpyDeviceToHost);
  /*
  for (uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputPrefixPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;
    it.thId = itemPtr[0];
    int16_t *ptr = (int16_t *)&(itemPtr[1]);
    it.endo = ptr[0] & 0x7FFF;
    it.mode = (ptr[0]&0x8000)!=0;
    it.incr = ptr[1];
    it.hash = (uint8_t *)(itemPtr + 2);
    prefixFound.push_back(it);
  }
  */
  for (uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputPrefixPinned + (i*ITEM_SIZE32);// #define ITEM_SIZE32 8
    ITEM it;
	it.thId = itemPtr[0];
    it.tpx = (uint32_t *)(itemPtr + 1);
	it.tkey = (uint32_t *)(itemPtr + 9);
	it.type = itemPtr[17];// 1 Tame 2 Wild
    prefixFound.push_back(it);
  }

  return callKernel();

}

/*
bool GPUEngine::CheckHash(uint8_t *h, vector<ITEM>& found,int tid,int incr,int endo, int *nbOK) {

  bool ok = true;

  // Search in found by GPU
  bool f = false;
  int l = 0;
  //printf("Search: %s\n", toHex(h,20).c_str());
  while (l < found.size() && !f) {
    f = ripemd160_comp_hash(found[l].hash, h);
    if (!f) l++;
  }
  if (f) {
    found.erase(found.begin() + l);
    *nbOK = *nbOK+1;
  } else {
    ok = false;
    printf("Expected item not found %s (thread=%d, incr=%d, endo=%d)\n",
      toHex(h, 20).c_str(),tid,incr,endo);
	if (found[l].hash != NULL)
		printf("%s\n", toHex(found[l].hash, 20).c_str());
	else
		printf("NULL\n");
  }

  return ok;

}
*/
