#ifndef __BIG_INT_H__
#define __BIG_INT_H__

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define __forceinline __attribute__((always_inline)) // added for linux

#if defined(_WIN32) || defined(_WIN64)
#ifdef USE_INT64_128
#error Cannot use __int128 on Windows 32 and 64 bit
#endif
#endif

// #ifdef USE_INT64_128
// #define MBITS		60
// // typedef unsigned __int64 BigInt;
// // typedef unsigned __int128 DBigInt;
// typedef uint64_t BigInt;
// typedef __int128 DBigInt;

// #else
// #define MBITS		28
// // typedef unsigned __int32 BigInt;
// // typedef unsigned __int64 DBigInt;
// typedef uint32_t BigInt;
// typedef uint64_t DBigInt;
// #endif


// #define MBITS		28 // 64 -> 60
// typedef uint32_t BigInt;
// typedef uint64_t DBigInt;

#define MBITS		60 // 64 -> 60
typedef uint64_t BigInt;
typedef __int128 DBigInt;

// In: <= DBigInt
// Out: decimal representation
// print <= DBigInt using printf
// static void bigintprint(char* str, DBigInt a) {
// 	printf("%s", str);
// 	printf("%" PRIu64 "\n", a);
// }

// In: <= DBigInt
// Out: binary representation with description
// recursively print binary bits
static void _bigintprintb(DBigInt num) {
    if (num >> 1) {
        _bigintprintb(num >> 1);
    }
    putc((num & 1) ? '1' : '0', stdout);
}

// host function for _bigintprintb
static void bigintprintb(char* str, DBigInt num) {
	printf("%s", str);
	_bigintprintb(num);
	printf("\n");
}

// get the number of decimal digits of $num
static int get_ndigits(BigInt num)
{
    int count = 0;
    while (num > 0)
    {
        count++;
        num /= 10;
    }
    return count;
}

static DBigInt karatsuba(BigInt a, BigInt b) {
    
    // base case
    // can set base case to 32-bit or 16-bit integers
    if (a < 10 && b < 10)
        return a * b;
    
    int ndigits = fmax(get_ndigits(a), get_ndigits(b));
    int nsplit = ceil(ndigits/2);

    BigInt a_high = a / (BigInt) pow(10, nsplit);
    BigInt a_low = a % (BigInt) pow(10, nsplit);
    BigInt b_high = b / (BigInt) pow(10, nsplit);
    BigInt b_low = b % (BigInt) pow(10, nsplit);

    // non-recursive method
    // DBigInt c1 = (DBigInt) a_high * (DBigInt) b_high;
	// DBigInt c2 = (DBigInt) a_low * (DBigInt) b_low;
	// DBigInt c3 = (DBigInt) (a_high + a_low) * (DBigInt) (b_high + b_low) - (c1 + c2);
	
    // recursive method
    DBigInt c1 = karatsuba(a_high, b_high);
    DBigInt c2 = karatsuba(a_low, b_low);
    DBigInt c3 = karatsuba(a_high + a_low, b_high + b_low) - (c1 + c2);

    return c1 * (DBigInt) pow(10, nsplit * 2) + c2 + c3 * (DBigInt) pow(10, nsplit);

}

static __forceinline  BigInt ModAddBigInt(BigInt a, BigInt b, BigInt m) {
	DBigInt a64 = a,
		b64 = b,
		m64 = m,
		c64 = a64 + b64,
		cm = c64 % m64;

	return (BigInt)cm;
}

static __forceinline  BigInt ModSubBigInt(BigInt a, BigInt b, BigInt m) {
	DBigInt a64 = a,
		b64 = b,
		m64 = m,
		c64 = (a64 < b64 ? a64 + m64 : a64) - b64,
		cm = c64 % m;

	return (BigInt)cm;
}

static __forceinline  BigInt ModMulBigInt(BigInt a, BigInt b, BigInt m) {
	DBigInt a64 = a,
		b64 = b,
		m64 = m,
		c64 = a64 * b64,
		cm = c64 % m64;

	return (BigInt)cm;
}

// * -> karatsuba
static __forceinline  BigInt ModMulKaratsubaBigInt(BigInt a, BigInt b, BigInt m) {
	DBigInt m64 = m,
		c64 = karatsuba(a, b),
		cm = c64 % m64; // still native mod

	return (BigInt)cm;
}


#if 0
static __forceinline  BigInt ModMulBarrettBigInt(BigInt a, BigInt b, BigInt m, BigInt mu) {
	DBigInt a64 = a,
		b64 = b,
		m64 = m,
		c64 = a64 * b64,
		cm = c64 % m64;

	return (BigInt)cm;
}
#else
// when using this functin, MBITS needs to match the data size
// e.g., if BigInt = uint64_t and actual data = uint32_t, MBITS needs to be 28
static __forceinline  BigInt ModMulBarrettBigInt(BigInt a, BigInt b, BigInt mod, BigInt mu) {
	DBigInt a64 = a,
		b64 = b,
		mod64 = mod,
		q64 = a64 * b64, // a
		tmp64 = q64, // a
		mu64 = mu,
		c64;
	q64 >>= (MBITS-2); // a >> bits-2
	// want: (a * 2^k/n >> 2^k) -> a/n
	q64 *= mu64; // (a >> bits-2) * mu
	q64 >>= (MBITS + 5); // (a >> 2*bits+3) * mu
	tmp64 -= q64 * mod64;
	c64 = tmp64 > mod64 ? tmp64 - mod64 : tmp64;

	return (BigInt) c64;
}
#endif

#endif
