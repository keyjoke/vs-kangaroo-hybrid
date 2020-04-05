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

#ifndef VANITYH
#define VANITYH

#include <string>
#include <vector>
#include "SECP256k1.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#endif

class VanitySearch;

/////////////////////////////////////////////////

typedef struct {
  VanitySearch *obj;
  int	threadId;
  bool	isRunning;
  bool	hasStarted;
  bool	rekeyRequest;
  int	gridSize;
  int	gpuId;

  Int	dK;
  Point Kp;
  bool	type; // false - Tame, true - Wild

} TH_PARAM;


typedef struct {
	Int bnL;
	Int bnU;
	Int bnW;
	int pow2L;
	int pow2U;
	int pow2W;
	Int bnM;
	Int bnWsqrt;
	int pow2Wsqrt;
} structW;

typedef struct {
	uint64_t n0;
	//uint64_t n1;
	//uint64_t n2;
	//uint64_t n3;
	Int distance;
} hashtb_entry;

typedef struct {

  char *prefix;
  int prefixLength;
  prefix_t sPrefix;
  double difficulty;
  bool *found;

  // For dreamer ;)
  bool isFull;
  //prefixl_t lPrefix;
  uint8_t hash160[20];
  
} PREFIX_ITEM;

typedef struct {

  std::vector<PREFIX_ITEM> *items;
  bool found;

} PREFIX_TABLE_ITEM;

/////////////////////////////////////////////////

class VanitySearch {

public:

  VanitySearch(Secp256K1 *secp, std::vector<std::string> &prefix, Point targetPubKey, structW *stR, int nbThread, int KangPower, bool stop, std::string outputFile, int flag_verbose, uint32_t maxFound, uint64_t rekey, bool flag_comparator);

  void Search(bool useGpu, std::vector<int> gpuId, std::vector<int> gridSize);
  void FindKeyCPU(TH_PARAM *p);
  void getGPUStartingKeys(int thId, int groupSize, int nbThread, Int *keys, Point *p, uint64_t *n_count);
  void getGPUStartingWKeys(int thId, int groupSize, int nbThread, Point w_targetPubKey, Int *w_keys, Point *w_p);// Wild start keys and Wild point
  bool File2save(Int px,  Int key, int stype);
  bool Comparator();
  void ReWriteFiles();
  bool TWSaveToDrive();
  bool TWUpload();
    
  void FindKeyGPU(TH_PARAM *p);
  void SolverGPU(TH_PARAM *p);

private:

  //std::string GetHex(std::vector<unsigned char> &buffer);
  bool checkPrivKeyCPU(Int &checkPrvKey, Point &pSample);
  
  //bool checkPrivKeyGPU(std::string addr, Int &key, int32_t incr, bool mode);
  //bool checkPrivKeyGPU(std::string addr, Int &key, int32_t incr, bool mode, uint32_t itThId, Int *greckey);
  //void checkAddr(int prefIdx, uint8_t *hash160, Int &key, int32_t incr, bool mode);
  //void checkAddr(int prefIdx, uint8_t *hash160, Int &key, int32_t incr, bool mode, uint32_t itThId, Int *reckey);

  bool output(std::string msg);
  bool outputgpu(std::string msg);

  bool isAlive(TH_PARAM *p);
  bool hasStarted(TH_PARAM *p);
  void rekeyRequest(TH_PARAM *p);
  uint64_t getGPUCount();
  uint64_t getCPUCount();
  //bool initPrefix(std::string &prefix, PREFIX_ITEM *it);
    
  //double getDiffuclty();
  //void updateFound();
  
  uint64_t getCountJ();

  uint64_t getJmaxofSp(Int& optimalmeanjumpsize, Int * dS);
    
  Secp256K1 *secp;
    
  Int key2;
  Int key3;
  Int wkey;
  uint32_t kadd_count;
      
  //int searchType;
  //int searchMode;
      
  Point	targetPubKey;
  Int resultPrvKey;

  uint64_t countj[256];

  int nbThread;
  int KangPower;
  bool TWRevers;
  bool flag_comparator;
  
  int nbCPUThread;
  int nbGPUThread;
  //int nbFoundKey;
  uint64_t rekey;
  uint64_t lastRekey;
  //uint32_t nbPrefix;
  std::string outputFile;
  //bool onlyFull;
  uint32_t maxFound;
  //double _difficulty;
  
  std::vector<PREFIX_TABLE_ITEM> prefixes;
  //std::vector<prefix_t> usedPrefix;
  //std::vector<LPREFIX> usedPrefixL;
  std::vector<std::string> &inputPrefixes;
 
  int flag_verbose;
  bool flag_endOfSearch;
  bool flag_startGPU;
    
  structW *stR;
  Int bnL, bnU, bnW;
  int pow2L, pow2U, pow2W;
  Int bnM, bnWsqrt;
  int pow2Wsqrt;

  int xU, xV;
  uint64_t xUV;
  Int bxU, bxV, bxUV;

  uint64_t DPmodule;
  uint64_t JmaxofSp;
  uint32_t GPUJmaxofSp;
  Int sizeJmax;

  uint64_t maxDP, countDT, countDW, countColl;
  uint64_t HASH_SIZE;
  hashtb_entry *DPht;

  char			buff_s32[32+1] = {0};
  unsigned char	buff_u32[32+1] = {0};

#ifdef WIN64
  HANDLE ghMutex;
#else
  pthread_mutex_t  ghMutex;
#endif

};

#endif // VANITYH
