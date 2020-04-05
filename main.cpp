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

#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include "default_pubkeys.h"

#define RELEASE "0.01"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("\nUsage:\n\n");
  printf("vs-kangaroo  [-h]\n");
  printf("             [-bits]\n");
  printf("             [-t cpucores]\n");
  
  printf("             [-gpu]\n");
  printf("             [-p Power]\n");
  printf("             [-cmp Comparator]\n");
  printf("             [-gpuId gpuId1[,gpuId2,...]]\n");
  printf("             [-g gridSize1[,gridSize2,...]]\n");
  printf("             [-l]\n");
  printf("             [-rekey]\n");

  printf("             [-o outputfile]\n");
  printf("             [-v level]\n");
  printf("             [PUBKEY]\n");
  printf("\n");

  printf(" PUBKEY ->Public key to recovery his private key\n");
  printf(" -bits 42 ->Range [2^41..2^42] search of keyspace\n");
  printf("       start:end ->Range [start..end] search of keyspace\n");
  printf(" -t 4 ->Num of CPU cores, must be even\n");
  
  printf(" -gpu ->Enable GPU calculation\n");
  printf(" -p 4 ->Num of GPU Kangaroo Power, must be 0-12 (default 0)\n");
  printf(" -cmp ->Enable Comparator Python, run solver.py (default C++)\n");
  printf(" -gpuId 0,1,2,.. ->List of GPUs to use, default is 0\n");
  printf(" -g 128,96,120,.. ->kernel gridsize, must be 8*(MP number)\n");
  printf(" -l ->List cuda enabled devices\n");
  printf(" -r rekey: Rekey interval in MegaJ, default is disabled \n");

  printf(" -o path2file ->Output found privkey to file\n");
  printf(" -v 1 ->Verbose level info\n");
  printf(" -h ->Usage\n");
      
  exit(EXIT_FAILURE);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("[FATAL_ERROR] Invalid %s argument, number expected\n",name.c_str());
    exit(EXIT_FAILURE);

  }

  return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens, const string &text, char sep) {

  size_t start = 0, end = 0;
  tokens.clear();
  int item;

  try {

    while ((end = text.find(sep, start)) != string::npos) {
      item = std::stoi(text.substr(start, end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  } catch(std::invalid_argument &) {

    printf("[FATAL_ERROR] Invalid %s argument, number expected\n",name.c_str());
    exit(EXIT_FAILURE);

  }

}

// ------------------------------------------------------------------------------------------
/*
bool CreateFileTargetPubKeyPrefix(Point &itargetPubKey, int ipow2bits) {
  
  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();
  
  Int step;
  step.SetInt32(1);
  // add G, 2G, 3G ... 8388608G ?
  //step.ShiftL(ipow2bits - 16); // 2^^110 = 2^^109 + 2^^109 = 2^^109 + (2^^96 * 8192) and 110 - 96 = 14 add 2 ?
  //printf("Target PubKey Prefix (file addr.txt) Key Step: 0x%s pow2bits - 16 \n", step.GetBase16().c_str());
  //printf("Target PubKey Prefix (file addr.txt) Key Step: 0x%s \n", step.GetBase16().c_str());
  Point pstep = secp->ComputePublicKey(&step);
  
  FILE *f = stdout;
  bool needToClose = false;
  if (ipow2bits > 34) {
    f = fopen("addr.txt", "wb");
    if (f == NULL) {
      printf("Cannot open addr.txt for writing \n");
      f = stdout;
	  return false;
    } else {
      needToClose = true;
    }
  } else {
	printf("Cannot created addr.txt pow2bits < 35 \n");
	return false;
  }
  
  for (int i = 0; i < 8388608; i++) { // 2^23
  
	
	std::string fprefix = itargetPubKey.x.GetBase16().c_str();
	std::string fgetstr;
	std::string str2 = "0";
	for (int is = (int)fprefix.size(); is < 64; is++) {
		fgetstr.append(str2);
	}
	fgetstr.append(fprefix);
	std::string fprefix20 = fgetstr.substr(0, 40);
	
	fprintf(f, "%s\n", fprefix20.c_str());
	
	itargetPubKey = secp->AddDirect(itargetPubKey, pstep);
	
  }
    
  if(needToClose)
	fclose(f);
    
  return true;
}
*/
// ------------------------------------------------------------------------------------------
/*
bool GenerateCode() {

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();
  
  // add
  int size = 1024;
  Int gstep;
  gstep.SetInt32(8388608);
  Point pgstep = secp->ComputePublicKey(&gstep);
  
  // Compute generator table
  Point *Gn = new Point[size];
  Point g = secp->G;
  Gn[0] = g;
  //g = secp->DoubleDirect(g);
  g = secp->AddDirect(g, pgstep);
  Gn[1] = g;
  for (int i = 2; i < size; i++) {
    // add
	//g = secp->AddDirect(g, secp->G);
	g = secp->AddDirect(g, pgstep);
    Gn[i] = g;
	printf("\nGenerate table GPUGroup_new.h i: %d", i);
  }
  // _2Gn = CPU_GRP_SIZE*G
  //Point _2Gn = secp->DoubleDirect(Gn[size / 2 - 1]);
  // add
  Int kGn;
  kGn.SetInt32(8388608);
  kGn.Mult(1024);
  Point pGn = secp->ComputePublicKey(&kGn);
  Point _2Gn = secp->AddDirect(Gn[0], pGn);
  //Point _2Gn = secp->DoubleDirect(gpGn);
    
  // Write file
  FILE *f = fopen("GPU/GPUGroup_new.h", "wb");

  fprintf(f, "// File generated by GPUEngine::GenerateCode()\n");
  fprintf(f, "// GROUP definitions\n");
  fprintf(f, "#define GRP_SIZE %d\n\n", size);
  fprintf(f, "// _2Gn = GRP_SIZE*G\n");
  fprintf(f, "__device__ __constant__ uint64_t _2Gnx[4] = %s;\n", _2Gn.x.GetC64Str(4).c_str());
  fprintf(f, "__device__ __constant__ uint64_t _2Gny[4] = %s;\n\n", _2Gn.y.GetC64Str(4).c_str());
  fprintf(f, "// SecpK1 Generator table (Contains G,8388609G,16777217G,25165825G...,(GRP_SIZE/2 )...G)\n");
  fprintf(f, "__device__ __constant__ uint64_t Gx[][4] = {\n");
  for (int i = 0; i < size/2; i++) {
    fprintf(f, "  %s,\n", Gn[i].x.GetC64Str(4).c_str());
  }
  fprintf(f, "};\n");

  fprintf(f, "__device__ __constant__ uint64_t Gy[][4] = {\n");
  for (int i = 0; i < size/2; i++) {
    fprintf(f, "  %s,\n", Gn[i].y.GetC64Str(4).c_str());
  }
  fprintf(f, "};\n\n");

  fclose(f);
  delete[] Gn;
  
  return true;
}
*/
// ------------------------------------------------------------------------------------------

bool GenerateSptable() {

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();
  
  int size = 512;
    
  // Compute generator table
  Point *Sp = new Point[size];
  Int *dS = new Int[size];
  Sp[0] = secp->G;
  dS[0].SetInt32(1);
  for (int i = 1; i < size; i++) {
	dS[i].Add(&dS[i-1], &dS[i-1]);
	Sp[i] = secp->DoubleAffine(Sp[i-1]);
	printf("\nGenerate Sp-table GPUSptable.h size i: %d", i);
  }
  
  // Write file
  FILE *f = fopen("GPU/GPUSptable.h", "wb");
  fprintf(f, "// File generated by Main::GenerateSptable()\n");
  fprintf(f, "#define GRP_SIZE %d\n\n", size);
  fprintf(f, "// SecpK1 Generator table (Contains G,2G,4G,8G...,(GRP_SIZE/2 )G)\n");
  fprintf(f, "__device__ __constant__ uint64_t Spx[][4] = {\n");
  for (int i = 0; i < size/2; i++) {
    fprintf(f, "  %s,\n", Sp[i].x.GetC64Str(4).c_str());
  }
  fprintf(f, "};\n");

  fprintf(f, "__device__ __constant__ uint64_t Spy[][4] = {\n");
  for (int i = 0; i < size/2; i++) {
    fprintf(f, "  %s,\n", Sp[i].y.GetC64Str(4).c_str());
  }
  fprintf(f, "};\n\n");
  
  fprintf(f, "__device__ __constant__ uint64_t dS[][4] = {\n");
  for (int i = 0; i < size/2; i++) {
    fprintf(f, "  %s,\n", dS[i].GetC64Str(4).c_str());
  }
  fprintf(f, "};\n\n");

  fclose(f);
  delete[] Sp;
  delete[] dS;
  
  return true;
}


// ------------------------------------------------------------------------------------------
/*
void parseFile(string fileName, vector<string> &lines) {
  
  // Get file size
  FILE *fp = fopen(fileName.c_str(), "rb");
  if (fp == NULL) {
    printf("Error: Cannot open %s %s\n", fileName.c_str(), strerror(errno));
    exit(-1);
  }
  fseek(fp, 0L, SEEK_END);
  size_t sz = ftell(fp);
  size_t nbAddr = sz / 33; // Upper approximation 
  bool loaddingProgress = sz > 100000;
  fclose(fp);

  // Parse file
  int nbLine = 0;
  string line;
  ifstream inFile(fileName);
  lines.reserve(nbAddr);
  while (getline(inFile, line)) {

    // Remove ending \r\n
    int l = (int)line.length() - 1;
    while (l >= 0 && isspace(line.at(l))) {
      line.pop_back();
      l--;
    }

    if (line.length() > 0) {
      lines.push_back(line);
      nbLine++;
      if (loaddingProgress) {
        if ((nbLine % 50000) == 0)
          printf("[Loading input file %5.1f%%]\r", ((double)nbLine*100.0) / ((double)(nbAddr)*33.0 / 34.0));
      }
    }

  }

  if (loaddingProgress)
    printf("[Loading input file 100.0%%]\n");

}
*/
// ------------------------------------------------------------------------------------------

void getKeySpace(const string &text, structW *stR, Int& maxKey) {

	size_t start = 0, end = 0;
	string item;

	try {

		if ((end = text.find(':', start)) != string::npos) {
			item = std::string(text.substr(start, end));
			start = end + 1;
		}
		else {
			item = std::string(text);
		}

		if (item.length() == 0) {
			stR->bnL.SetInt32(1);
		}
		else if (item.length() > 64) {
			printf("[FATAL_ERROR] START invalid (64 length)\n");
			exit(EXIT_FAILURE);
		}
		else {
			item.insert(0, 64 - item.length(), '0');
			for (int i = 0; i < 32; ++i) {
				unsigned char my1ch = 0;
				sscanf(&item[2 * i], "%02hhX", &my1ch);
				stR->bnL.SetByte(31 - i, my1ch);
			}
		}
		//printf("[keyspaceSTART] 0x%064s \n", stR->bnL.GetBase16().c_str());

		if (start != 0 && (end = text.find('+', start)) != string::npos) {
			item = std::string(text.substr(end + 1));
			if (item.length() > 64 || item.length() == 0) {
				printf("\n[FATAL_ERROR] END invalid (64 length)\n");
				exit(EXIT_FAILURE);
			}
			item.insert(0, 64 - item.length(), '0');
			for (int i = 0; i < 32; ++i) {
				unsigned char my1ch = 0;
				sscanf(&item[2 * i], "%02hhX", &my1ch);
				stR->bnU.SetByte(31 - i, my1ch);
			}
			stR->bnU.Add(&stR->bnL);
		}
		else if (start != 0) {
			item = std::string(text.substr(start));
			if (item.length() > 64 || item.length() == 0) {
				printf("\n[FATAL_ERROR] END invalid (64 length)\n");
				exit(EXIT_FAILURE);
			}
			item.insert(0, 64 - item.length(), '0');
			for (int i = 0; i < 32; ++i) {
				unsigned char my1ch = 0;
				sscanf(&item[2 * i], "%02hhX", &my1ch);
				stR->bnU.SetByte(31 - i, my1ch);
			}
		}
		else {
			stR->bnU.Set(&maxKey);
		}
		//printf("[keyspace__END] 0x%064s \n", stR->bnU.GetBase16().c_str());

	}
	catch (std::invalid_argument &) {

		printf("\n[FATAL_ERROR] Invalid --keyspace argument \n");
		exit(EXIT_FAILURE);

	}

}

// ------------------------------------------------------------------------------------------

void checkKeySpace(structW *stR, Int& maxKey) {

	if (stR->bnL.IsGreater(&maxKey) || stR->bnU.IsGreater(&maxKey)) {
		printf("\n[ERROR] START/END IsGreater %064s \n", maxKey.GetBase16().c_str());
		exit(EXIT_FAILURE);
	}
	if (stR->bnU.IsLowerOrEqual(&stR->bnL)) {
		printf("\n[ERROR] END IsLowerOrEqual START \n");
		exit(EXIT_FAILURE);
	}

	return;
}

// ------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  setbuf(stdout, NULL); // cancel channel buffering (for printf)

  // Global Init
  Timer::Init();
  rseed((unsigned long)time(NULL));

  // Init SecpK1
  Secp256K1 *secp = new Secp256K1();
  secp->Init();

  printf("[###########################################################]\n");
  printf("[#          Pollard-kangaroo PrivKey Recovery Tool         #]\n");
  printf("[#          (based on engine of VanitySearch 1.15)         #]\n");
  printf("[#                 bitcoin ecdsa secp256k1                 #]\n");
  printf("[#                 ver %4s GPU Hybrid v4F                 #]\n", RELEASE);
  printf("[#          GPU implementation changes by alek76           #]\n");
  printf("[#          Tips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9       #]\n");
  printf("[###########################################################]\n");

  time_t timenow = time(NULL);
  //char *timebuff;timebuff = ctime(&timenow);
  char timebuff[255+1]; strftime(timebuff, sizeof(timebuff), "%d %b %Y %H:%M:%S", gmtime(&timenow));
  printf("[DATE(utc)] %s\n", timebuff);

  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");


  // Browse arguments
  if (argc < 1) {
    printf("\n[FATAL_ERROR] Not enough argument\n");
    printUsage();
  }

  /////////////////////////////////////////////////

  int a = 1; // count args

  bool useGpu = false;
  bool stop = true;
  //int searchMode = SEARCH_COMPRESSED;
  vector<int> gpuId = {0};
  vector<int> gridSize = {-1};
  vector<string> prefix;

  int nbCPUThread = Timer::getCoreNumber();
  bool tSpecified = false;
  
  //uint32_t maxFound = 65536;
  uint32_t maxFound = 131072;
  uint64_t rekey = 0;
  int GPUPower = 0;
  bool flag_cmp = false;

  string outputFile = "Result.txt";

  int	flag_verbose = 0;
  bool	flag_compressed;

  Point targetPubKey;
  targetPubKey.Clear();

 int pow2bits = 0;
 int bitsMin = 12;
 int bitsMax = 160;

 uint64_t DPmodule = 0;

 // max distinguished points in hashtable
 //uint64_t maxDP = 1 << 10; // 2^10=1024

 structW structRange, *stR;
 stR = &structRange;
 stR->bnL.SetInt32(0);
 stR->bnU.SetInt32(0);
 stR->bnW.SetInt32(0);
 stR->pow2L = 0;
 stR->pow2U = 0;
 stR->pow2W = 0;
 stR->bnM.SetInt32(0);
 stR->bnWsqrt.SetInt32(0);
 stR->pow2Wsqrt = 0;

 Int maxKey;
 maxKey.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");

 Int Ktmp;
 
 /////////////////////////////////////////////////

  while (a < argc) {

    if (strcmp(argv[a], "-h")==0 || strcmp(argv[a], "-help")==0 || strcmp(argv[a], "--help")==0 || strcmp(argv[a], "/?")==0) {
	  printUsage();
	}
	else if (strcmp(argv[a], "-v") == 0) {
	  a++;
	  flag_verbose = getInt("pow2bits", argv[a]);
	  a++;
	}
	else if (strcmp(argv[a], "-l") == 0) {
	  #ifdef WITHGPU
	  GPUEngine::PrintCudaInfo();
	  #else
	  printf("\n[DEBUG] GPU code not compiled, use -DWITHGPU when compiling.\n");
	  #endif
	  exit(0);
	}
	else if (strcmp(argv[a], "-gpu") == 0) {
	  useGpu = true;
	  a++;
	}
	else if (strcmp(argv[a], "-gpuId") == 0) {
	  a++;
	  getInts("gpuId", gpuId, string(argv[a]), ',');
	  a++;
    }
	else if (strcmp(argv[a], "-g") == 0) {
      a++;
      getInts("gridSize", gridSize, string(argv[a]),',');
      a++;
    }
	else if (strcmp(argv[a], "-o") == 0) {
      a++;
      outputFile = string(argv[a]);
      a++;
    }
	else if (strcmp(argv[a], "-t") == 0) {
      a++;
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      a++;
      tSpecified = true;
	}
	else if (strcmp(argv[a], "-p") == 0) {
      a++;
      GPUPower = getInt("GPUPower",argv[a]);
      a++;      
	}
	else if (strcmp(argv[a], "-cmp") == 0) {
      flag_cmp = true;
      a++;	  
	}
	else if (strcmp(argv[a], "-bits") == 0) {
	//} else if (strcmp(argv[a], "--keyspace") == 0) {
	  a++;
	  if(string(argv[a]).find(':')!=std::string::npos){
		  // L, U
		  getKeySpace(string(argv[a]), stR, maxKey);
		  checkKeySpace(stR, maxKey);

		  // W
		  stR->bnW.Sub(&stR->bnU, &stR->bnL);

		  // pow2L
		  Ktmp = stR->bnL;
		  for (int i = 0 ; i < 256 ; ++i) {
			  Ktmp.ShiftR(1);
			  stR->pow2L = i+0;
			  if (Ktmp.IsZero()) break;
		  }

		  // pow2U
		  Ktmp = stR->bnU;
		  for (int i = 0; i < 256; ++i) {
			  Ktmp.ShiftR(1);
			  stR->pow2U = i+1;
			  if (Ktmp.IsZero()) break;
		  }

		  // pow2W
		  Ktmp = stR->bnW;
		  for (int i = 0; i < 256; ++i) {
			  Ktmp.ShiftR(1);
			  stR->pow2W = i+0;
			  if (Ktmp.IsZero()) break;
		  }

		  printf("[L] (2^%u) %064s\n", stR->pow2L, stR->bnL.GetBase16().c_str());
		  printf("[U] (2^%u) %064s\n", stR->pow2U, stR->bnU.GetBase16().c_str());
		  printf("[W] (2^%u) %064s\n", stR->pow2W, stR->bnW.GetBase16().c_str());

		  pow2bits = stR->pow2U;

	  }
	  else {
		  pow2bits = getInt("pow2bits", argv[a]);

		  if (pow2bits<0) pow2bits = 0;
		  if (!pow2bits && useGpu)	pow2bits = 65;
		  if (!pow2bits && !useGpu)	pow2bits = 42;
		  printf("[pow2bits]	%u\n", pow2bits);

		  stR->pow2L = pow2bits - 1;
		  stR->pow2U = pow2bits - 0;
		  stR->pow2W = pow2bits - 1;

		  // L
		  stR->bnL.SetInt32(1);
		  stR->bnL.ShiftL(stR->pow2L);
		  
		  // U
		  stR->bnU.SetInt32(1);
		  stR->bnU.ShiftL(stR->pow2U);
		  
		  // W
		  stR->bnW.Sub(&stR->bnU, &stR->bnL);

	  }
	  a++;
	} 
	else if (strcmp(argv[a], "-r") == 0) {
      a++;
      rekey = (uint64_t)getInt("rekey", argv[a]);
      a++;
    } 
	else if (a == argc - 1) {
	  targetPubKey = secp->ParsePublicKeyHex(string(argv[a]), flag_compressed);
	  /*
	  //add
	  std::string aprefix = targetPubKey.x.GetBase16().c_str();
	  std::string getstr;
	  std::string str1 = "0";
	  for (int i = aprefix.size(); i < 64; i++) {
		getstr.append(str1);
	  }
	  getstr.append(aprefix);
	  //printf("\nLen: %d \n", getstr.size());
	  std::string aprefix20 = getstr.substr(0, 40);
	  //printf("\nPrefix argument: %s \n", aprefix20.c_str());
	  prefix.push_back(aprefix20);
	  */
	  a++;
    } 
	else {
      printf("\n[FATAL_ERROR] Unexpected %s argument\n",argv[a]);
      printUsage();
    }

  }
  
  // Generate Sp-table GPUSptable.h
  //bool genSp = GenerateSptable();
  
  // Create File addr.txt
  //if (useGpu){
	//Point pp = targetPubKey;
	//printf("Create File addr.txt (328MB) wait... \n");
	//bool fprefix = CreateFileTargetPubKeyPrefix(pp, pow2bits);
	//bool gt = GenerateCode();// Generate table GPUGroup_new.h
	//if (fprefix){
	//	printf("\nFile addr.txt - Ready\n");
	//} else {
	//	printf("\n[FATAL_ERROR] Create File addr.txt \n");
	//	exit(EXIT_FAILURE);
	//}
  //Timer::SleepMillis(2000);
  //parseFile("addr.txt",prefix);
  //}
  //

  /////////////////////////////////////////////////
  // some checks

  // min <= bits <= max
  if (stR->pow2W < bitsMin || bitsMax < stR->pow2W) {
	  printf("\n[FATAL_ERROR] -bits must be %u..%u \n", bitsMin, bitsMax);
	  exit(EXIT_FAILURE);
  }
  if (stR->pow2W > 70) {
	  printf("[warning!] bits = 2^%u too big! long runtime expected\n", stR->pow2W);
  }
  
  // Wsqrt
  int round = (int)stR->pow2W / 2;//int round = (float)stR->pow2W / 2;
  stR->pow2Wsqrt = round;
  //stR->pow2Wsqrt = (int)round((float)stR->pow2W / 2);
  //stR->bnWsqrt.SetInt32(1); bnWsqrt.ShiftL(stR->pow2Wsqrt); // rude
  stR->bnWsqrt = stR->bnW; for (int i = 0; i < stR->pow2W / 2; ++i) { stR->bnWsqrt.ShiftR(1); }; // floor															  
  printf("[Wsqrt] (2^%u) %064s\n", stR->pow2W / 2, stR->bnWsqrt.GetBase16().c_str());

  // M = (L+U)/2 == L+(W/2)
  stR->bnM.Add(&stR->bnL, &stR->bnU);
  stR->bnM.ShiftR(1);
  printf("[M] %064s\n", stR->bnM.GetBase16().c_str());


  // load pubkey from default table
  string pubkeyhex;
  if (targetPubKey.isZero()) {
	  for (int i = 0; i < 33; ++i) { char pub[2 + 1]; sprintf(pub, "%02hhX", default_pubkeys[stR->pow2U][i]); pubkeyhex += string(pub); }
	  //printf("[pubkeyhex] %s\n", pubkeyhex.c_str());
	  targetPubKey = secp->ParsePublicKeyHex(pubkeyhex, flag_compressed);
  }

  //grid
  if(gpuId.size()!=gridSize.size()) {
    if(gridSize.size()==1 && gridSize[0]==-1) {
      gridSize.clear();
      for(int i=0;i<gpuId.size();++i)
        gridSize.push_back(-1);
    }
	else {
      printf("\n[error] Invalid gridSize or gpuId argument, must have same size\n");
      printUsage();
    }
  }

  //th
  if (nbCPUThread<0) nbCPUThread = Timer::getCoreNumber();
  if (nbCPUThread % 2) nbCPUThread -= 1; // number cpu_cores must be even
  if (useGpu) nbCPUThread = 0;
  if (!useGpu && nbCPUThread == 0) printUsage();    
  
  /////////////////////////////////////////////////
  // run

  //VanitySearch *v = new VanitySearch(secp, targetPubKey, stR, nbCPUThread, outputFile, flag_verbose);
  VanitySearch *v = new VanitySearch(secp, prefix, targetPubKey, stR, nbCPUThread, GPUPower, stop, outputFile, flag_verbose, maxFound, rekey, flag_cmp);
  
  v->Search(useGpu, gpuId, gridSize);

  return 0;
}
