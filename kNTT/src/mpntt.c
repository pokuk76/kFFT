// Compile with: gcc -std=c99 -lm mpntt.c -o mpntt
// Run with: ./mpntt

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

// NTTX DBaseInt methods
#include "big_int.h"

#define N 8
#define TWDGEN 0

#define MBITS 60
typedef uint16_t HBaseInt;
typedef uint32_t BaseInt;
typedef uint64_t DBaseInt;
typedef __int128 QBaseInt; // should be only used for testing
#define BaseIntType PRIu32

#define BaseSize (int) sizeof(BaseInt)*8

// ----------------- BEGIN of SPIRAL-generated code -----------------

/*
 * This code was generated by Spiral 8.5.0, www.spiral.net
 */

#include <stdint.h>
#include "big_int.h"

void init_ntt8bigint() {
}

void ntt8bigint(BigInt  *Y, BigInt  *X, BigInt modulus, BigInt  *twiddles) {
    BigInt s73, s74, s75, s76, s77, s78, s79, s80, 
            s81, s82, s83, s84, t87, t88, t89, t90, 
            t91, t92, t93, t94;
    s73 = ModMulBigInt(twiddles[1], X[4], modulus);
    t87 = ModAddBigInt(X[0], s73, modulus); // c = a + b % m
    t88 = ModSubBigInt(X[0], s73, modulus); // c = a - b % m
    s74 = ModMulBigInt(twiddles[1], X[5], modulus);
    t89 = ModAddBigInt(X[1], s74, modulus);
    t90 = ModSubBigInt(X[1], s74, modulus);
    s75 = ModMulBigInt(twiddles[1], X[6], modulus);
    s76 = ModMulBigInt(twiddles[2], ModAddBigInt(X[2], s75, modulus), modulus);
    s77 = ModMulBigInt(twiddles[3], ModSubBigInt(X[2], s75, modulus), modulus);
    s78 = ModMulBigInt(twiddles[1], X[7], modulus);
    s79 = ModMulBigInt(twiddles[2], ModAddBigInt(X[3], s78, modulus), modulus);
    s80 = ModMulBigInt(twiddles[3], ModSubBigInt(X[3], s78, modulus), modulus);
    t91 = ModAddBigInt(t87, s76, modulus);
    t92 = ModSubBigInt(t87, s76, modulus);
    t93 = ModAddBigInt(t88, s77, modulus);
    t94 = ModSubBigInt(t88, s77, modulus);
    s81 = ModMulBigInt(twiddles[4], ModAddBigInt(t89, s79, modulus), modulus);
    s82 = ModMulBigInt(twiddles[5], ModSubBigInt(t89, s79, modulus), modulus);
    s83 = ModMulBigInt(twiddles[6], ModAddBigInt(t90, s80, modulus), modulus);
    s84 = ModMulBigInt(twiddles[7], ModSubBigInt(t90, s80, modulus), modulus);
    Y[0] = ModAddBigInt(t91, s81, modulus);
    Y[1] = ModSubBigInt(t91, s81, modulus);
    Y[2] = ModAddBigInt(t92, s82, modulus);
    Y[3] = ModSubBigInt(t92, s82, modulus);
    Y[4] = ModAddBigInt(t93, s83, modulus);
    Y[5] = ModSubBigInt(t93, s83, modulus);
    Y[6] = ModAddBigInt(t94, s84, modulus);
    Y[7] = ModSubBigInt(t94, s84, modulus);
}

void destroy_ntt8bigint() {
}

// ----------------- END of SPIRAL-generated code -----------------


// ----------------- host code for verification -----------------

void verify(DBaseInt * truth, DBaseInt * obtained){
    for (int i = 0; i < N; i++){
        if (obtained[i] != truth[i]) {
            printf("Error at %d\n", i);
            return;
        }
    }
    printf("\nVerified!\n");
}


int main(){

    DBaseInt y[N];

    // Q: [1, W^0, W^0, W^2, W^0, W^2, W^1, W^3]
    // N: 8 32b
    DBaseInt mu = 2147484408; // not needed
    DBaseInt modulus = 268435361;
    DBaseInt twd[N] = {1, 106416730, 210605896, 90281519, 18125476, 56427457, 86117821, 123453994};
    DBaseInt x[N] = {66180002, 46054340, 185133904, 68728779, 156137740, 135212474, 87317267, 204961903};
    DBaseInt y_t[N] = {177566360, 54723412, 49777885, 162151521, 82033502, 102309405, 212936481, 224812172};

    printf("BaseSize: %d\n", BaseSize);
    printf("Testing %d-point MP NTTs using %d-bit DBaseInt on CPU...\n", N, BaseSize*2);

    // Executing kernel 
    ntt8bigint(y, x, modulus, twd);

    // Verification
    verify(y_t, y);

}