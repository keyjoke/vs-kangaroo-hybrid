/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *da
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

// CUDA Kernel main function
// Compute SecpK1 keys and calculate RIPEMD160(SHA256(key)) then check prefix
// For the kernel, we use a 16 bits prefix lookup table which correspond to ~3 Base58 characters
// A second level lookup table contains 32 bits prefix (if used)
// (The CPU computes the full address and check the full prefix)
// 
// We use affine coordinates for elliptic curve point (ie Z=1)

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void SendPoint(uint64_t *pointx, uint64_t *distance, uint32_t type, uint32_t maxFound, uint32_t *out) {
	
	uint64_t OutPx[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t OutKey[4] = { 0ULL,0ULL,0ULL,0ULL };
		
	uint32_t pos;
	uint32_t tid = (blockIdx.x*blockDim.x) + threadIdx.x;
		
	Load256(OutPx, pointx);
	Load256(OutKey, distance);
		
	// add Item
	pos = atomicAdd(out, 1);
	if (pos < maxFound) {
		// ITEM_SIZE32 8
		out[pos*ITEM_SIZE32 + 0] = tid;
		out[pos*ITEM_SIZE32 + 1] = (uint32_t)(OutPx[0]);
		out[pos*ITEM_SIZE32 + 2] = (uint32_t)(OutPx[0] >> 32);
		out[pos*ITEM_SIZE32 + 3] = (uint32_t)(OutPx[1]);
		out[pos*ITEM_SIZE32 + 4] = (uint32_t)(OutPx[1] >> 32);
		out[pos*ITEM_SIZE32 + 5] = (uint32_t)(OutPx[2]);
		out[pos*ITEM_SIZE32 + 6] = (uint32_t)(OutPx[2] >> 32);
		out[pos*ITEM_SIZE32 + 7] = (uint32_t)(OutPx[3]);
		out[pos*ITEM_SIZE32 + 8] = (uint32_t)(OutPx[3] >> 32);		
		out[pos*ITEM_SIZE32 + 9] = (uint32_t)(OutKey[0]);
		out[pos*ITEM_SIZE32 + 10] = (uint32_t)(OutKey[0] >> 32);
		out[pos*ITEM_SIZE32 + 11] = (uint32_t)(OutKey[1]);
		out[pos*ITEM_SIZE32 + 12] = (uint32_t)(OutKey[1] >> 32);
		out[pos*ITEM_SIZE32 + 13] = (uint32_t)(OutKey[2]);
		out[pos*ITEM_SIZE32 + 14] = (uint32_t)(OutKey[2] >> 32);
		out[pos*ITEM_SIZE32 + 15] = (uint32_t)(OutKey[3]);
		out[pos*ITEM_SIZE32 + 16] = (uint32_t)(OutKey[3] >> 32);		
		out[pos*ITEM_SIZE32 + 17] = type;// 1 Tame 2 Wild
	}	
}

#define SetPoint(pointx, distance, type) SendPoint(pointx, distance, type, maxFound, out);

// -----------------------------------------------------------------------------------------

__device__ void CheckDP(uint64_t *check_px, uint64_t *check_wpx, uint64_t *check_tk, uint64_t *check_wk, uint32_t type, uint64_t DPmodule, uint32_t maxFound, uint32_t *out) {
	
	// For Check
	//uint64_t outPx[4] = {0x4444444444444444ULL, 0x3333333333333333ULL, 0x2222222222222222ULL, 0x1111111111111111ULL};
	//uint64_t outKey[4] = {0x8888888888888888ULL, 0x7777777777777777ULL, 0x6666666666666666ULL, 0x5555555555555555ULL};
	//type = 2;// Wild
	//type = 1;// Tame
		
	uint64_t tame_px[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t wild_px[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	Load256(tame_px, check_px);
	Load256(wild_px, check_wpx);
	
	
	if (type == 1) {// Tame 1
		if (tame_px[0] % DPmodule == 0) {
			//SetPoint(outPx, outKey, type);
			SetPoint(check_px, check_tk, type);
		}
	}
	
	if (type == 2) {// Wild 2
		if (wild_px[0] % DPmodule == 0) {
			//SetPoint(outPx, outKey, type);
			SetPoint(check_wpx, check_wk, type);
		}
	}
	
}

#define CHECK_POINT(check_px, check_wpx, check_tk, check_wk, type) CheckDP(check_px, check_wpx, check_tk, check_wk, type, DPmod, maxFound, out)

// -----------------------------------------------------------------------------------------

__device__ void ComputeKeys(uint64_t *startx, uint64_t *starty, uint64_t *wstartx, uint64_t *wstarty, uint64_t *TStartKey, uint64_t *WStartKey, 
                            uint64_t DPmod, uint32_t hop_modulo, uint32_t maxFound, uint32_t *out) {

	// wstartx, wstarty - Wild point	
	uint64_t dx[GRP_SIZE/2+1][4];
	uint64_t wdx[GRP_SIZE/2+1][4];
		
	uint64_t px[4] = { 0ULL,0ULL,0ULL,0ULL };// Tame point
	uint64_t py[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t wpx[4] = { 0ULL,0ULL,0ULL,0ULL };// Wild point
	uint64_t wpy[4] = { 0ULL,0ULL,0ULL,0ULL };
		
	uint64_t sx[4] = { 0ULL,0ULL,0ULL,0ULL };// Tame point
	uint64_t sy[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t wsx[4] = { 0ULL,0ULL,0ULL,0ULL };// Wild point
	uint64_t wsy[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	uint64_t tsk[4] = { 0ULL,0ULL,0ULL,0ULL };// Tame start key
	uint64_t wsk[4] = { 0ULL,0ULL,0ULL,0ULL };// Wild start key
	uint64_t tk[4] = { 0ULL,0ULL,0ULL,0ULL };// Tame key
	uint64_t wk[4] = { 0ULL,0ULL,0ULL,0ULL };// Wild key
	
	uint32_t type = 0;
	uint32_t pw2 = 0;// Wild
	uint32_t pw1 = 0;// Tame
		
	uint64_t tempWPx[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t tempPx[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	uint64_t dy[4] = { 0ULL,0ULL,0ULL,0ULL };// Tame
	uint64_t _s[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t _p2[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	uint64_t wdy[4] = { 0ULL,0ULL,0ULL,0ULL };// Wild
	uint64_t w_s[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t w_p2[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	// Load starting key
	__syncthreads();
	Load256A(sx, startx);
	Load256A(sy, starty);
	Load256(px, sx);
	Load256(py, sy);
		
	// Wild point	
	Load256A(wsx, wstartx);
	Load256A(wsy, wstarty);
	Load256(wpx, wsx);
	Load256(wpy, wsy);
		
	// Tame and Wild start key	
	Load256A(tsk, TStartKey);
	Load256A(wsk, WStartKey);
	Load256(tk, tsk);
	Load256(wk, wsk);
		
	// Sp-table
	/*
	uint32_t i;
	for (i = 0; i < HSIZE; i++){
		ModSub256(dx[i], Spx[i], sx);
		ModSub256(wdx[i], Spx[i], wsx);
	}
	ModSub256(dx[i], Spx[i], sx);
	ModSub256(wdx[i], Spx[i], wsx);
	ModSub256(dx[i+1], Spx[hop_modulo], sx);// For the next point
	ModSub256(wdx[i+1], Spx[hop_modulo], wsx);
		
	// Compute modular inverse
	_ModInvGrouped(dx);
	_ModInvGrouped(wdx);
	*/
	
	uint32_t j;
	for (j = 0; j < hop_modulo; j++) { 
	
		// check, is it distinguished point ?
		
		if (1) {// Wild
			
			type = 2;// Wild 2
			
			// uint64_t pw = ph->Kp.x.bits64[0] % JmaxofSp;
			Load256(tempWPx, wpx);
			pw2 = tempWPx[0] % hop_modulo;
			
			
			// nowjumpsize = 1 << pw
			//Int nowjumpsize = dS[pw];
			//ph->dK.Add(&nowjumpsize);
			
			// Add Hops Distance Wild
			//Load256(wk, wsk);
			ModAdd256(wk, dS[pw2]);
			
			
			// Affine points addition
			// ph->Kp = secp->AddAffine(ph->Kp, Sp[pw]);
			
			//Load256(wpx, wsx);
			//Load256(wpy, wsy);
			ModSub256(wdy, Spy[pw2], wpy);
			
			// Get wdx
			ModSub256(wdx[pw2], Spx[pw2], wpx);
			_ModInvNoGrouped(wdx[pw2]);
			
			_ModMult(w_s, wdy, wdx[pw2]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
			_ModSqr(w_p2, w_s);               // _p2 = pow2(s)
			
			ModSub256(wpx, w_p2, wpx);
			ModSub256(wpx, Spx[pw2]);          // px = pow2(s) - p1.x - p2.x;
			
			ModSub256(wpy, Spx[pw2], wpx);
			_ModMult(wpy, w_s);               // py = - s*(ret.x-p2.x)
			ModSub256(wpy, Spy[pw2]);          // py = - p2.y - s*(ret.x-p2.x);
			
			
			CHECK_POINT(px, wpx, tk, wk, type);
			
		}
		
		if (1){// Tame
			
			type = 1;// Tame 1
			
			// uint64_t pw = ph->Kp.x.bits64[0] % JmaxofSp;
			Load256(tempPx, px);			
			pw1 = tempPx[0] % hop_modulo;
			
			
			// nowjumpsize = 1 << pw
			// Int nowjumpsize = dS[pw];
			// ph->dK.Add(&nowjumpsize);
			
			// Add Hops Distance Tame
			//Load256(tk, tsk);
			ModAdd256(tk, dS[pw1]);
			
			
			// Affine points addition
			// ph->Kp = secp->AddAffine(ph->Kp, Sp[pw]);
			
			//Load256(px, sx);
			//Load256(py, sy);
			ModSub256(dy, Spy[pw1], py);
			
			// Get dx
			ModSub256(dx[pw1], Spx[pw1], px);
			_ModInvNoGrouped(dx[pw1]);
			
			_ModMult(_s, dy, dx[pw1]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
			_ModSqr(_p2, _s);              // _p2 = pow2(s)
			
			ModSub256(px, _p2, px);
			ModSub256(px, Spx[pw1]);        // px = pow2(s) - p1.x - p2.x;
			
			ModSub256(py, Spx[pw1], px);
			_ModMult(py, _s);              // py = - s*(ret.x-p2.x)
			ModSub256(py, Spy[pw1]);        // py = - p2.y - s*(ret.x-p2.x);
			
			
			CHECK_POINT(px, wpx, tk, wk, type);			
			
		}
		
	}
	
	// Update starting point
	__syncthreads();
	Store256A(startx, px);
	Store256A(starty, py);
	
	Store256A(wstartx, wpx);
	Store256A(wstarty, wpy);
	
	Store256A(TStartKey, tk);
	Store256A(WStartKey, wk);	
}
