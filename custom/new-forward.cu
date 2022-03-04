#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#define USE_OPT 9
#define TILE_SIZE 32 // for mat mul only
#define SEC 2 // for unroll only
// MT2 gpu baseline
// 0: use None 

// MT3 optimizations
// 1: high dimensional shared memory
// 2: constant mem
const int kernelLength1 = 1 * 7 * 7 * 4;
const int kernelLength2 = 4 * 7 * 7 * 16;
__constant__ float Mc1[kernelLength1];
__constant__ float Mc2[kernelLength2];


constexpr int tile_width1 = 16;
constexpr int tile_width2 = 16;
constexpr int K1=7;
constexpr int K2=7;
constexpr int C1=1;
constexpr int C2=4;

// 3 loop unrolling and restrict keyword
// 4  Half Precision
typedef __half half_t;
// 5: 1d shared memory
// 6: different layer different scheme

// Final Submission - mat-mul
// 7: baseline mat-mul: shared memory matrix multiplication and input matrix unrolling
// 8: constant memory for conv filter as an opt
const int kernelLengthUnroll1 = ((4-1)/TILE_SIZE+1)*TILE_SIZE*((K1*K1*C1-1)/TILE_SIZE+1)*TILE_SIZE;
const int kernelLengthUnroll2 = ((16-1)/TILE_SIZE+1)*TILE_SIZE*((K2*K2*C2-1)/TILE_SIZE+1)*TILE_SIZE;
__constant__ float Mc1Unroll[kernelLengthUnroll1];
__constant__ float Mc2Unroll[kernelLengthUnroll2];
// 9: built in unrolling



// unroll only the input feature maps; the weights are already unrolled by default
__global__ void unroll_kernel(float* X_unroll, const float *x, const int B, const int C, const int H, const int W, const int K)
{
	int H_out = H - K + 1;
    int W_out = W - K + 1;
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    int c = blockIdx.x; // input feature map parallel
    int by = blockIdx.y; // linearized output 2d block
    int b = blockIdx.z; // batch parallel
	
	int dx = blockDim.x;
	int dy = blockDim.y;

	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	int w = (by%w_grid)*dx+tx; // output x 
    int h = (by/w_grid)*dy+ty; // output y 
	int w_base = c*K*K;
	#define X_unroll3d(i2, i1, i0) X_unroll[i2*(W_out*H_out*K*K*C) + i1*(W_out*H_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

	// each thread is responsible for only K^2 elements
	if(h<H_out&&w<W_out){
		for(int p=0;p<K;p++)
		for(int q=0;q<K;q++){
			int h_unroll= w_base + p * K + q;
			int w_unroll= h * W_out+ w;
			X_unroll3d(b, h_unroll, w_unroll) = x4d(b, c, h + p, w + q); // copy input pixels	
		}
	}
	#undef X_unroll3d
	#undef x4d
}

__global__ void conv_forward_matmul_kernel(const float *A, const float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
{
	__shared__ float subTileA[TILE_SIZE][TILE_SIZE];
	__shared__ float subTileB[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int b = blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by*TILE_SIZE+ty;
	int Col = bx*TILE_SIZE+tx;
	float Pvalue = 0;
	
	for(int q=0;q<(numAColumns-1)/TILE_SIZE+1;q++){
		if(Row<numARows && (q*TILE_SIZE+tx)<numAColumns ) subTileA[ty][tx]=A[Row*numAColumns+q*TILE_SIZE+tx];
		else subTileA[ty][tx]=0;
		if(Col<numBColumns && (q*TILE_SIZE+ty)<numBRows) subTileB[ty][tx]=B[b*(numBRows*numBColumns)+(q*TILE_SIZE+ty)*numBColumns+Col];
		else subTileB[ty][tx]=0;
		__syncthreads();
		if(Row<numCRows && Col<numCColumns)
		for(int k=0;k<TILE_SIZE;k++){
			Pvalue+=subTileA[ty][k]*subTileB[k][tx];
		}
		__syncthreads();
	}
	if(Row<numCRows && Col<numCColumns)
		C[b*numCRows*numCColumns+Row*numCColumns+Col]=Pvalue;
}

__global__ void conv_forward_matmul_constmemory_kernel1(const float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
{
	__shared__ float subTileB[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int b = blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by*TILE_SIZE+ty;
	int Col = bx*TILE_SIZE+tx;
	float Pvalue = 0;
	int offset1 = Row*numAColumns;
	int offset2 = 0;
	
	for(int q=0;q<(numAColumns-1)/TILE_SIZE+1;q++){
		if(Col<numBColumns && (q*TILE_SIZE+ty)<numBRows) subTileB[ty][tx]=B[b*(numBRows*numBColumns)+(q*TILE_SIZE+ty)*numBColumns+Col];
		else subTileB[ty][tx]=0;
		offset2 = q*TILE_SIZE;
		__syncthreads();
		if(Row<numCRows && Col<numCColumns)
		for(int k=0;k<TILE_SIZE;k++){
			Pvalue+=Mc1Unroll[offset1+offset2+k]*subTileB[k][tx];
		}
		__syncthreads();
	}
	if(Row<numCRows && Col<numCColumns)
		C[b*numCRows*numCColumns+Row*numCColumns+Col]=Pvalue;
}
__global__ void conv_forward_matmul_constmemory_kernel2(const float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
{
	__shared__ float subTileB[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int b = blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by*TILE_SIZE+ty;
	int Col = bx*TILE_SIZE+tx;
	float Pvalue = 0;
	int offset1 = Row*numAColumns;
	int offset2 = 0;
	
	for(int q=0;q<(numAColumns-1)/TILE_SIZE+1;q++){
		if(Col<numBColumns && (q*TILE_SIZE+ty)<numBRows) subTileB[ty][tx]=B[b*(numBRows*numBColumns)+(q*TILE_SIZE+ty)*numBColumns+Col];
		else subTileB[ty][tx]=0;
		offset2 = q*TILE_SIZE;
		__syncthreads();
		if(Row<numCRows && Col<numCColumns)
		for(int k=0;k<TILE_SIZE;k++){
			Pvalue+=Mc2Unroll[offset1+offset2+k]*subTileB[k][tx];
		}
		__syncthreads();
	}
	if(Row<numCRows && Col<numCColumns)
		C[b*numCRows*numCColumns+Row*numCColumns+Col]=Pvalue;
}

__global__ void conv_forward_matmul_builtin_unroll_kernel (float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	int num_W_Unroll_Rows = M;
	int num_W_Unroll_Columns = K*K*C;
	int num_X_Unroll_Rows = K*K*C;
	int num_X_Unroll_Columns = W_out*H_out;
	int num_Y_Unroll_Rows = M;
	int num_Y_Unroll_Columns = W_out*H_out;
	
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]	

	__shared__ float subTileA[TILE_SIZE][TILE_SIZE];
	__shared__ float subTileB[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int b = blockIdx.z;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by*TILE_SIZE+ty;
	int Col = bx*TILE_SIZE+tx;
	float Pvalue = 0;
	
	for(int q=0;q<(num_W_Unroll_Columns-1)/TILE_SIZE+1;q++){
		if(Row<num_W_Unroll_Rows && (q*TILE_SIZE+tx)<num_W_Unroll_Columns ) subTileA[ty][tx]=k[Row*num_W_Unroll_Columns+q*TILE_SIZE+tx];
		else subTileA[ty][tx]=0;
		if(Col<num_X_Unroll_Columns && (q*TILE_SIZE+ty)<num_X_Unroll_Rows) subTileB[ty][tx]=x4d(b, (q*TILE_SIZE+ty)/(K*K), Col/W_out+(q*TILE_SIZE+ty)%(K*K)/K, Col%W_out+(q*TILE_SIZE+ty)%(K*K)%K);
		else subTileB[ty][tx]=0;
		__syncthreads();
		if(Row<num_Y_Unroll_Rows && Col<num_Y_Unroll_Columns)
		for(int k=0;k<TILE_SIZE;k++){
			Pvalue+=subTileA[ty][k]*subTileB[k][tx];
		}
		__syncthreads();
	}
	#undef x4d

	if(Row<num_Y_Unroll_Rows && Col<num_Y_Unroll_Columns)
		y[b*num_Y_Unroll_Rows*num_Y_Unroll_Columns+Row*num_Y_Unroll_Columns+Col]=Pvalue;
}

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
	
	int dx = blockDim.x;
	int dy = blockDim.y;
	int dz = blockDim.z;
	
	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	
	int m,h,w,c,b,p,q;
	float acc;
	m=bx;
	h=(by/w_grid)*dy+ty;
	w=(by%w_grid)*dx+tx;
	b=bz;
	if((w<W_out)&&(h<H_out)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
		y4d(b,m,h,w)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}


__global__ void Float2HalfKernel(half_t* Output, const float* Input, const int length) 
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int dx = blockDim.x;

    int l = bx * dx + tx;

    if ( l < length) {
        Output[l] = __float2half(Input[l]);
    }
}

__global__ void Half2FloatKernel(float* Output, const half_t* Input, const int length) 
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int dx = blockDim.x;

    int l = bx * dx + tx;

    if ( l < length) {
        Output[l] = __half2float(Input[l]);
    }
}
__global__ void conv4_forward_kernel(half_t *y, const half_t *x, const half_t *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
	
	int dx = blockDim.x;
	int dy = blockDim.y;
	int dz = blockDim.z;
	
	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	
	int m,h,w,c,b,p,q;
	half_t acc;
	m=bx;
	h=(by/w_grid)*dy+ty;
	w=(by%w_grid)*dx+tx;
	b=bz;
	if((w<W_out)&&(h<H_out)){
		acc=0;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc= __hadd(acc,__hmul(x4d(b,c,h+p,w+q),k4d(m,c,p,q)));
		y4d(b,m,h,w)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}

__global__ void conv31_forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
	
	int dx = blockDim.x;
	int dy = blockDim.y;
	int dz = blockDim.z;
	
	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	
	int m,h,w,c,b,p,q;
	float acc;
	m=bx;
	h=(by/w_grid)*dy+ty;
	w=(by%w_grid)*dx+tx;
	b=bz;
	if((w<W_out)&&(h<H_out)){
		acc=0.0f;
		acc+=x4d(b,0,h+0,w+0)*k4d(m,0,0,0)+x4d(b,0,h+0,w+1)*k4d(m,0,0,1)+x4d(b,0,h+0,w+2)*k4d(m,0,0,2)+x4d(b,0,h+0,w+3)*k4d(m,0,0,3)+x4d(b,0,h+0,w+4)*k4d(m,0,0,4)+x4d(b,0,h+0,w+5)*k4d(m,0,0,5)+x4d(b,0,h+0,w+6)*k4d(m,0,0,6)
+x4d(b,0,h+1,w+0)*k4d(m,0,1,0)+x4d(b,0,h+1,w+1)*k4d(m,0,1,1)+x4d(b,0,h+1,w+2)*k4d(m,0,1,2)+x4d(b,0,h+1,w+3)*k4d(m,0,1,3)+x4d(b,0,h+1,w+4)*k4d(m,0,1,4)+x4d(b,0,h+1,w+5)*k4d(m,0,1,5)+x4d(b,0,h+1,w+6)*k4d(m,0,1,6)
+x4d(b,0,h+2,w+0)*k4d(m,0,2,0)+x4d(b,0,h+2,w+1)*k4d(m,0,2,1)+x4d(b,0,h+2,w+2)*k4d(m,0,2,2)+x4d(b,0,h+2,w+3)*k4d(m,0,2,3)+x4d(b,0,h+2,w+4)*k4d(m,0,2,4)+x4d(b,0,h+2,w+5)*k4d(m,0,2,5)+x4d(b,0,h+2,w+6)*k4d(m,0,2,6)
+x4d(b,0,h+3,w+0)*k4d(m,0,3,0)+x4d(b,0,h+3,w+1)*k4d(m,0,3,1)+x4d(b,0,h+3,w+2)*k4d(m,0,3,2)+x4d(b,0,h+3,w+3)*k4d(m,0,3,3)+x4d(b,0,h+3,w+4)*k4d(m,0,3,4)+x4d(b,0,h+3,w+5)*k4d(m,0,3,5)+x4d(b,0,h+3,w+6)*k4d(m,0,3,6)
+x4d(b,0,h+4,w+0)*k4d(m,0,4,0)+x4d(b,0,h+4,w+1)*k4d(m,0,4,1)+x4d(b,0,h+4,w+2)*k4d(m,0,4,2)+x4d(b,0,h+4,w+3)*k4d(m,0,4,3)+x4d(b,0,h+4,w+4)*k4d(m,0,4,4)+x4d(b,0,h+4,w+5)*k4d(m,0,4,5)+x4d(b,0,h+4,w+6)*k4d(m,0,4,6)
+x4d(b,0,h+5,w+0)*k4d(m,0,5,0)+x4d(b,0,h+5,w+1)*k4d(m,0,5,1)+x4d(b,0,h+5,w+2)*k4d(m,0,5,2)+x4d(b,0,h+5,w+3)*k4d(m,0,5,3)+x4d(b,0,h+5,w+4)*k4d(m,0,5,4)+x4d(b,0,h+5,w+5)*k4d(m,0,5,5)+x4d(b,0,h+5,w+6)*k4d(m,0,5,6)
+x4d(b,0,h+6,w+0)*k4d(m,0,6,0)+x4d(b,0,h+6,w+1)*k4d(m,0,6,1)+x4d(b,0,h+6,w+2)*k4d(m,0,6,2)+x4d(b,0,h+6,w+3)*k4d(m,0,6,3)+x4d(b,0,h+6,w+4)*k4d(m,0,6,4)+x4d(b,0,h+6,w+5)*k4d(m,0,6,5)+x4d(b,0,h+6,w+6)*k4d(m,0,6,6);
		y4d(b,m,h,w)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}

__global__ void conv32_forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
	
	int dx = blockDim.x;
	int dy = blockDim.y;
	int dz = blockDim.z;
	
	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	
	int m,h,w,c,b,p,q;
	float acc;
	m=bx;
	h=(by/w_grid)*dy+ty;
	w=(by%w_grid)*dx+tx;
	b=bz;
	if((w<W_out)&&(h<H_out)){
		acc=0.0f;
		acc+=x4d(b,0,h+0,w+0)*k4d(m,0,0,0)+x4d(b,0,h+0,w+1)*k4d(m,0,0,1)+x4d(b,0,h+0,w+2)*k4d(m,0,0,2)+x4d(b,0,h+0,w+3)*k4d(m,0,0,3)+x4d(b,0,h+0,w+4)*k4d(m,0,0,4)+x4d(b,0,h+0,w+5)*k4d(m,0,0,5)+x4d(b,0,h+0,w+6)*k4d(m,0,0,6)
+x4d(b,0,h+1,w+0)*k4d(m,0,1,0)+x4d(b,0,h+1,w+1)*k4d(m,0,1,1)+x4d(b,0,h+1,w+2)*k4d(m,0,1,2)+x4d(b,0,h+1,w+3)*k4d(m,0,1,3)+x4d(b,0,h+1,w+4)*k4d(m,0,1,4)+x4d(b,0,h+1,w+5)*k4d(m,0,1,5)+x4d(b,0,h+1,w+6)*k4d(m,0,1,6)
+x4d(b,0,h+2,w+0)*k4d(m,0,2,0)+x4d(b,0,h+2,w+1)*k4d(m,0,2,1)+x4d(b,0,h+2,w+2)*k4d(m,0,2,2)+x4d(b,0,h+2,w+3)*k4d(m,0,2,3)+x4d(b,0,h+2,w+4)*k4d(m,0,2,4)+x4d(b,0,h+2,w+5)*k4d(m,0,2,5)+x4d(b,0,h+2,w+6)*k4d(m,0,2,6)
+x4d(b,0,h+3,w+0)*k4d(m,0,3,0)+x4d(b,0,h+3,w+1)*k4d(m,0,3,1)+x4d(b,0,h+3,w+2)*k4d(m,0,3,2)+x4d(b,0,h+3,w+3)*k4d(m,0,3,3)+x4d(b,0,h+3,w+4)*k4d(m,0,3,4)+x4d(b,0,h+3,w+5)*k4d(m,0,3,5)+x4d(b,0,h+3,w+6)*k4d(m,0,3,6)
+x4d(b,0,h+4,w+0)*k4d(m,0,4,0)+x4d(b,0,h+4,w+1)*k4d(m,0,4,1)+x4d(b,0,h+4,w+2)*k4d(m,0,4,2)+x4d(b,0,h+4,w+3)*k4d(m,0,4,3)+x4d(b,0,h+4,w+4)*k4d(m,0,4,4)+x4d(b,0,h+4,w+5)*k4d(m,0,4,5)+x4d(b,0,h+4,w+6)*k4d(m,0,4,6)
+x4d(b,0,h+5,w+0)*k4d(m,0,5,0)+x4d(b,0,h+5,w+1)*k4d(m,0,5,1)+x4d(b,0,h+5,w+2)*k4d(m,0,5,2)+x4d(b,0,h+5,w+3)*k4d(m,0,5,3)+x4d(b,0,h+5,w+4)*k4d(m,0,5,4)+x4d(b,0,h+5,w+5)*k4d(m,0,5,5)+x4d(b,0,h+5,w+6)*k4d(m,0,5,6)
+x4d(b,0,h+6,w+0)*k4d(m,0,6,0)+x4d(b,0,h+6,w+1)*k4d(m,0,6,1)+x4d(b,0,h+6,w+2)*k4d(m,0,6,2)+x4d(b,0,h+6,w+3)*k4d(m,0,6,3)+x4d(b,0,h+6,w+4)*k4d(m,0,6,4)+x4d(b,0,h+6,w+5)*k4d(m,0,6,5)+x4d(b,0,h+6,w+6)*k4d(m,0,6,6);

		acc+=x4d(b,1,h+0,w+0)*k4d(m,1,0,0)+x4d(b,1,h+0,w+1)*k4d(m,1,0,1)+x4d(b,1,h+0,w+2)*k4d(m,1,0,2)+x4d(b,1,h+0,w+3)*k4d(m,1,0,3)+x4d(b,1,h+0,w+4)*k4d(m,1,0,4)+x4d(b,1,h+0,w+5)*k4d(m,1,0,5)+x4d(b,1,h+0,w+6)*k4d(m,1,0,6)
+x4d(b,1,h+1,w+0)*k4d(m,1,1,0)+x4d(b,1,h+1,w+1)*k4d(m,1,1,1)+x4d(b,1,h+1,w+2)*k4d(m,1,1,2)+x4d(b,1,h+1,w+3)*k4d(m,1,1,3)+x4d(b,1,h+1,w+4)*k4d(m,1,1,4)+x4d(b,1,h+1,w+5)*k4d(m,1,1,5)+x4d(b,1,h+1,w+6)*k4d(m,1,1,6)
+x4d(b,1,h+2,w+0)*k4d(m,1,2,0)+x4d(b,1,h+2,w+1)*k4d(m,1,2,1)+x4d(b,1,h+2,w+2)*k4d(m,1,2,2)+x4d(b,1,h+2,w+3)*k4d(m,1,2,3)+x4d(b,1,h+2,w+4)*k4d(m,1,2,4)+x4d(b,1,h+2,w+5)*k4d(m,1,2,5)+x4d(b,1,h+2,w+6)*k4d(m,1,2,6)
+x4d(b,1,h+3,w+0)*k4d(m,1,3,0)+x4d(b,1,h+3,w+1)*k4d(m,1,3,1)+x4d(b,1,h+3,w+2)*k4d(m,1,3,2)+x4d(b,1,h+3,w+3)*k4d(m,1,3,3)+x4d(b,1,h+3,w+4)*k4d(m,1,3,4)+x4d(b,1,h+3,w+5)*k4d(m,1,3,5)+x4d(b,1,h+3,w+6)*k4d(m,1,3,6)
+x4d(b,1,h+4,w+0)*k4d(m,1,4,0)+x4d(b,1,h+4,w+1)*k4d(m,1,4,1)+x4d(b,1,h+4,w+2)*k4d(m,1,4,2)+x4d(b,1,h+4,w+3)*k4d(m,1,4,3)+x4d(b,1,h+4,w+4)*k4d(m,1,4,4)+x4d(b,1,h+4,w+5)*k4d(m,1,4,5)+x4d(b,1,h+4,w+6)*k4d(m,1,4,6)
+x4d(b,1,h+5,w+0)*k4d(m,1,5,0)+x4d(b,1,h+5,w+1)*k4d(m,1,5,1)+x4d(b,1,h+5,w+2)*k4d(m,1,5,2)+x4d(b,1,h+5,w+3)*k4d(m,1,5,3)+x4d(b,1,h+5,w+4)*k4d(m,1,5,4)+x4d(b,1,h+5,w+5)*k4d(m,1,5,5)+x4d(b,1,h+5,w+6)*k4d(m,1,5,6)
+x4d(b,1,h+6,w+0)*k4d(m,1,6,0)+x4d(b,1,h+6,w+1)*k4d(m,1,6,1)+x4d(b,1,h+6,w+2)*k4d(m,1,6,2)+x4d(b,1,h+6,w+3)*k4d(m,1,6,3)+x4d(b,1,h+6,w+4)*k4d(m,1,6,4)+x4d(b,1,h+6,w+5)*k4d(m,1,6,5)+x4d(b,1,h+6,w+6)*k4d(m,1,6,6);

		acc+=x4d(b,2,h+0,w+0)*k4d(m,2,0,0)+x4d(b,2,h+0,w+1)*k4d(m,2,0,1)+x4d(b,2,h+0,w+2)*k4d(m,2,0,2)+x4d(b,2,h+0,w+3)*k4d(m,2,0,3)+x4d(b,2,h+0,w+4)*k4d(m,2,0,4)+x4d(b,2,h+0,w+5)*k4d(m,2,0,5)+x4d(b,2,h+0,w+6)*k4d(m,2,0,6)
+x4d(b,2,h+1,w+0)*k4d(m,2,1,0)+x4d(b,2,h+1,w+1)*k4d(m,2,1,1)+x4d(b,2,h+1,w+2)*k4d(m,2,1,2)+x4d(b,2,h+1,w+3)*k4d(m,2,1,3)+x4d(b,2,h+1,w+4)*k4d(m,2,1,4)+x4d(b,2,h+1,w+5)*k4d(m,2,1,5)+x4d(b,2,h+1,w+6)*k4d(m,2,1,6)
+x4d(b,2,h+2,w+0)*k4d(m,2,2,0)+x4d(b,2,h+2,w+1)*k4d(m,2,2,1)+x4d(b,2,h+2,w+2)*k4d(m,2,2,2)+x4d(b,2,h+2,w+3)*k4d(m,2,2,3)+x4d(b,2,h+2,w+4)*k4d(m,2,2,4)+x4d(b,2,h+2,w+5)*k4d(m,2,2,5)+x4d(b,2,h+2,w+6)*k4d(m,2,2,6)
+x4d(b,2,h+3,w+0)*k4d(m,2,3,0)+x4d(b,2,h+3,w+1)*k4d(m,2,3,1)+x4d(b,2,h+3,w+2)*k4d(m,2,3,2)+x4d(b,2,h+3,w+3)*k4d(m,2,3,3)+x4d(b,2,h+3,w+4)*k4d(m,2,3,4)+x4d(b,2,h+3,w+5)*k4d(m,2,3,5)+x4d(b,2,h+3,w+6)*k4d(m,2,3,6)
+x4d(b,2,h+4,w+0)*k4d(m,2,4,0)+x4d(b,2,h+4,w+1)*k4d(m,2,4,1)+x4d(b,2,h+4,w+2)*k4d(m,2,4,2)+x4d(b,2,h+4,w+3)*k4d(m,2,4,3)+x4d(b,2,h+4,w+4)*k4d(m,2,4,4)+x4d(b,2,h+4,w+5)*k4d(m,2,4,5)+x4d(b,2,h+4,w+6)*k4d(m,2,4,6)
+x4d(b,2,h+5,w+0)*k4d(m,2,5,0)+x4d(b,2,h+5,w+1)*k4d(m,2,5,1)+x4d(b,2,h+5,w+2)*k4d(m,2,5,2)+x4d(b,2,h+5,w+3)*k4d(m,2,5,3)+x4d(b,2,h+5,w+4)*k4d(m,2,5,4)+x4d(b,2,h+5,w+5)*k4d(m,2,5,5)+x4d(b,2,h+5,w+6)*k4d(m,2,5,6)
+x4d(b,2,h+6,w+0)*k4d(m,2,6,0)+x4d(b,2,h+6,w+1)*k4d(m,2,6,1)+x4d(b,2,h+6,w+2)*k4d(m,2,6,2)+x4d(b,2,h+6,w+3)*k4d(m,2,6,3)+x4d(b,2,h+6,w+4)*k4d(m,2,6,4)+x4d(b,2,h+6,w+5)*k4d(m,2,6,5)+x4d(b,2,h+6,w+6)*k4d(m,2,6,6);
		
		acc+=x4d(b,3,h+0,w+0)*k4d(m,3,0,0)+x4d(b,3,h+0,w+1)*k4d(m,3,0,1)+x4d(b,3,h+0,w+2)*k4d(m,3,0,2)+x4d(b,3,h+0,w+3)*k4d(m,3,0,3)+x4d(b,3,h+0,w+4)*k4d(m,3,0,4)+x4d(b,3,h+0,w+5)*k4d(m,3,0,5)+x4d(b,3,h+0,w+6)*k4d(m,3,0,6)
+x4d(b,3,h+1,w+0)*k4d(m,3,1,0)+x4d(b,3,h+1,w+1)*k4d(m,3,1,1)+x4d(b,3,h+1,w+2)*k4d(m,3,1,2)+x4d(b,3,h+1,w+3)*k4d(m,3,1,3)+x4d(b,3,h+1,w+4)*k4d(m,3,1,4)+x4d(b,3,h+1,w+5)*k4d(m,3,1,5)+x4d(b,3,h+1,w+6)*k4d(m,3,1,6)
+x4d(b,3,h+2,w+0)*k4d(m,3,2,0)+x4d(b,3,h+2,w+1)*k4d(m,3,2,1)+x4d(b,3,h+2,w+2)*k4d(m,3,2,2)+x4d(b,3,h+2,w+3)*k4d(m,3,2,3)+x4d(b,3,h+2,w+4)*k4d(m,3,2,4)+x4d(b,3,h+2,w+5)*k4d(m,3,2,5)+x4d(b,3,h+2,w+6)*k4d(m,3,2,6)
+x4d(b,3,h+3,w+0)*k4d(m,3,3,0)+x4d(b,3,h+3,w+1)*k4d(m,3,3,1)+x4d(b,3,h+3,w+2)*k4d(m,3,3,2)+x4d(b,3,h+3,w+3)*k4d(m,3,3,3)+x4d(b,3,h+3,w+4)*k4d(m,3,3,4)+x4d(b,3,h+3,w+5)*k4d(m,3,3,5)+x4d(b,3,h+3,w+6)*k4d(m,3,3,6)
+x4d(b,3,h+4,w+0)*k4d(m,3,4,0)+x4d(b,3,h+4,w+1)*k4d(m,3,4,1)+x4d(b,3,h+4,w+2)*k4d(m,3,4,2)+x4d(b,3,h+4,w+3)*k4d(m,3,4,3)+x4d(b,3,h+4,w+4)*k4d(m,3,4,4)+x4d(b,3,h+4,w+5)*k4d(m,3,4,5)+x4d(b,3,h+4,w+6)*k4d(m,3,4,6)
+x4d(b,3,h+5,w+0)*k4d(m,3,5,0)+x4d(b,3,h+5,w+1)*k4d(m,3,5,1)+x4d(b,3,h+5,w+2)*k4d(m,3,5,2)+x4d(b,3,h+5,w+3)*k4d(m,3,5,3)+x4d(b,3,h+5,w+4)*k4d(m,3,5,4)+x4d(b,3,h+5,w+5)*k4d(m,3,5,5)+x4d(b,3,h+5,w+6)*k4d(m,3,5,6)
+x4d(b,3,h+6,w+0)*k4d(m,3,6,0)+x4d(b,3,h+6,w+1)*k4d(m,3,6,1)+x4d(b,3,h+6,w+2)*k4d(m,3,6,2)+x4d(b,3,h+6,w+3)*k4d(m,3,6,3)+x4d(b,3,h+6,w+4)*k4d(m,3,6,4)+x4d(b,3,h+6,w+5)*k4d(m,3,6,5)+x4d(b,3,h+6,w+6)*k4d(m,3,6,6);

		y4d(b,m,h,w)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}
__global__ void conv21_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	__shared__ float x_data[C1][tile_width1 + K1 - 1][tile_width1 + K1 - 1];
	
	
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int m,h,w,c,b,p,q;
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;

	int w_grid= ceil(W_out/(1.0 * tile_width1));
	int h_grid= ceil(H_out/(1.0 * tile_width1));

	int w_o = (by%w_grid)*tile_width1+tx; // global pixel x 
    int h_o = (by/w_grid)*tile_width1+ty; // global pixel y 

    int w_i = w_o;// - K/2;
    int h_i = h_o;// - K/2;
	
    //int z_i = z_o - MASK_RADIUS;
	m=bx; // output feature map
	b=bz; //  batch

	for(c=0;c<C1;c++){
		if (w_i >= 0 && h_i >= 0 && w_i < W && h_i < H)
			x_data[c][ty][tx] = x4d(b,c,h_i,w_i);
		else
			x_data[c][ty][tx] = 0.0f;
	}
    __syncthreads();
	
	
	float acc;
	
	if((w_o<W_out)&&(h_o<H_out)&&(tx<tile_width1)&&(ty<tile_width1)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=x_data[c][ty+p][tx+q]*k4d(m,c,p,q);
		y4d(b,m,h_o,w_o)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}
__global__ void conv22_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	__shared__ float x_data[C2][tile_width2 + K2 - 1][tile_width2 + K2 - 1];
	
	
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int m,h,w,c,b,p,q;
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;

	int w_grid= ceil(W_out/(1.0 * tile_width2));
	int h_grid= ceil(H_out/(1.0 * tile_width2));

	int w_o = (by%w_grid)*tile_width2+tx; // global pixel x 
    int h_o = (by/w_grid)*tile_width2+ty; // global pixel y 

    int w_i = w_o;// - K/2;
    int h_i = h_o;// - K/2;
	
    //int z_i = z_o - MASK_RADIUS;
	m=bx; // output feature map
	b=bz; //  batch

	for(c=0;c<C2;c++){
		if (w_i >= 0 && h_i >= 0 && w_i < W && h_i < H)
			x_data[c][ty][tx] = x4d(b,c,h_i,w_i);
		else
			x_data[c][ty][tx] = 0.0f;
	}
    __syncthreads();
	
	
	float acc;
	
	if((w_o<W_out)&&(h_o<H_out)&&(tx<tile_width2)&&(ty<tile_width2)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=x_data[c][ty+p][tx+q]*k4d(m,c,p,q);
		y4d(b,m,h_o,w_o)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}

__global__ void conv51_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	__shared__ float x_data[C1*(tile_width1 + K1 - 1)*(tile_width1 + K1 - 1)];
	
	
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define shared3d(i2, i1, i0) x_data[(i2) * ((tile_width1 + K1 - 1) * (tile_width1 + K1 - 1)) + (i1) * (tile_width1 + K1 - 1) + i0]
    // Insert your GPU convolution kernel code here
	int m,h,w,c,b,p,q;
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;

	int w_grid= ceil(W_out/(1.0 * tile_width1));
	int h_grid= ceil(H_out/(1.0 * tile_width1));

	int w_o = (by%w_grid)*tile_width1+tx; // global pixel x 
    int h_o = (by/w_grid)*tile_width1+ty; // global pixel y 

    int w_i = w_o;// - K/2;
    int h_i = h_o;// - K/2;
	
    //int z_i = z_o - MASK_RADIUS;
	m=bx; // output feature map
	b=bz; //  batch

	for(c=0;c<C1;c++){
		if (w_i >= 0 && h_i >= 0 && w_i < W && h_i < H)
			shared3d(c,ty,tx) = x4d(b,c,h_i,w_i);
		else
			shared3d(c,ty,tx) = 0.0f;
	}
    __syncthreads();
	
	
	float acc;
	
	if((w_o<W_out)&&(h_o<H_out)&&(tx<tile_width1)&&(ty<tile_width1)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=shared3d(c,ty+p,tx+q)*k4d(m,c,p,q);
		y4d(b,m,h_o,w_o)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
#undef shared3d
}
__global__ void conv52_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	__shared__ float x_data[C2*(tile_width2 + K2 - 1)*(tile_width2 + K2 - 1)];
	
	
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define shared3d(i2, i1, i0) x_data[(i2) * ((tile_width2 + K2 - 1) * (tile_width2 + K2 - 1)) + (i1) * (tile_width2 + K2 - 1) + i0]

    // Insert your GPU convolution kernel code here
	int m,h,w,c,b,p,q;
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;

	int w_grid= ceil(W_out/(1.0 * tile_width2));
	int h_grid= ceil(H_out/(1.0 * tile_width2));

	int w_o = (by%w_grid)*tile_width2+tx; // global pixel x 
    int h_o = (by/w_grid)*tile_width2+ty; // global pixel y 

    int w_i = w_o;// - K/2;
    int h_i = h_o;// - K/2;
	
    //int z_i = z_o - MASK_RADIUS;
	m=bx; // output feature map
	b=bz; //  batch

	for(c=0;c<C2;c++){
		if (w_i >= 0 && h_i >= 0 && w_i < W && h_i < H)
			shared3d(c,ty,tx) = x4d(b,c,h_i,w_i);
		else
			shared3d(c,ty,tx) = 0.0f;
	}
    __syncthreads();
	
	
	float acc;
	
	if((w_o<W_out)&&(h_o<H_out)&&(tx<tile_width2)&&(ty<tile_width2)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=shared3d(c,ty+p,tx+q)*k4d(m,c,p,q);
		y4d(b,m,h_o,w_o)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
#undef shared3d
}

__global__ void conv11_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
	
	int dx = blockDim.x;
	int dy = blockDim.y;
	int dz = blockDim.z;
	
	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	
	int m,h,w,c,b,p,q;
	float acc;
	m=bx;
	h=(by/w_grid)*dy+ty;
	w=(by%w_grid)*dx+tx;
	b=bz;
	if((w<W_out)&&(h<H_out)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
		y4d(b,m,h,w)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}
__global__ void conv12_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tz = threadIdx.z;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
	
	int dx = blockDim.x;
	int dy = blockDim.y;
	int dz = blockDim.z;
	
	int w_grid= ceil(W_out/(1.0 * dx));
	int h_grid= ceil(H_out/(1.0 * dy));
	
	int m,h,w,c,b,p,q;
	float acc;
	m=bx;
	h=(by/w_grid)*dy+ty;
	w=(by%w_grid)*dx+tx;
	b=bz;
	if((w<W_out)&&(h<H_out)){
		acc=0.0f;
		for(c=0;c<C;c++)
			for(p=0;p<K;p++)
				for(q=0;q<K;q++)
					acc+=x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
		y4d(b,m,h,w)=acc;
	}
#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
	int h_out = H-K+1;
	int w_out = W-K+1;
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.


	
	cudaMalloc((void**)(device_x_ptr), (C * H * W * B) * sizeof(float));
	cudaMalloc((void**)(device_y_ptr), (M * h_out * w_out * B) * sizeof(float));
	cudaMalloc((void**)(device_k_ptr), (C * K * K * M) * sizeof(float));
	//cudaMemset((*device_y_ptr), 0, (M * h_out * w_out * B) * sizeof(float));
	cudaMemcpy(*device_x_ptr, host_x, (C * H * W * B) * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(*device_k_ptr, host_k, (C * K * K * M) * sizeof(float),cudaMemcpyHostToDevice);
	
	
	//@@ Define constant memory for device kernel here
	float* host_unroll_padding;
	if(C==C1){
		if(USE_OPT!=8) cudaMemcpyToSymbol(Mc1, host_k, kernelLength1 * sizeof(float));
		else{
			int rows_padding = ((M-1)/TILE_SIZE+1)*TILE_SIZE;
			int cols_padding = ((K*K*C-1)/TILE_SIZE+1)*TILE_SIZE;
			host_unroll_padding = (float*)malloc(rows_padding*cols_padding*sizeof(float));
			int cols_before_padding = K*K*C;
			for(int i=0;i<rows_padding;i++){
				for(int j=0;j<cols_padding;j++){
					if(i<M && j<cols_before_padding) host_unroll_padding[i*cols_padding+j]=host_k[i*cols_before_padding+j];
					else host_unroll_padding[i*cols_padding+j]=0;
				}
			}
			cudaMemcpyToSymbol(Mc1Unroll, host_unroll_padding, kernelLengthUnroll1 * sizeof(float));
		}
	}
	else if(C==C2){
		if(USE_OPT!=8) cudaMemcpyToSymbol(Mc2, host_k, kernelLength2 * sizeof(float));
		else{
			int rows_padding = ((M-1)/TILE_SIZE+1)*TILE_SIZE;
			int cols_padding = ((K*K*C-1)/TILE_SIZE+1)*TILE_SIZE;
			host_unroll_padding = (float*)malloc(rows_padding*cols_padding*sizeof(float));
			int cols_before_padding = K*K*C;
			for(int i=0;i<rows_padding;i++){
				for(int j=0;j<cols_padding;j++){
					if(i<M && j<cols_before_padding) host_unroll_padding[i*cols_padding+j]=host_k[i*cols_before_padding+j];
					else host_unroll_padding[i*cols_padding+j]=0;
				}
			}
			cudaMemcpyToSymbol(Mc2Unroll, host_unroll_padding, kernelLengthUnroll2 * sizeof(float));
		}
	}
	
	
    //Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
	{
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	
    // Set the kernel dimensions and call the kernel
	int h_out = H-K+1;
	int w_out = W-K+1;
	int w_grid;  // number of horizontal tiles per output map
	int h_grid; // number of vertical tiles per output map
	int Y;  // 2d to 1d
	dim3 dimGrid;
	dim3 dimBlock;

	if(USE_OPT==0){
		int w_grid = ceil(w_out/(1.0 * tile_width1)); // number of horizontal tiles per output map
		int h_grid = ceil(h_out/(1.0 * tile_width1)); // number of vertical tiles per output map
		Y = h_grid*w_grid; // 2d to 1d
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width1, tile_width1, 1);
		conv_forward_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, device_k, B, M, C, H, W, K);
	}
	else if(C==C1 && USE_OPT==1){
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width1, tile_width1, 1);
		conv11_forward_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, B, M, C, H, W, K);
	}
	else if(C==C2 && USE_OPT==1){
		w_grid = ceil(w_out/(1.0 * tile_width2));
		h_grid = ceil(h_out/(1.0 * tile_width2));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width2, tile_width2, 1);
		conv12_forward_kernel <<<dimGrid, dimBlock>>> (device_y, device_x, B, M, C, H, W, K);
	}
	else if(C==C1 && USE_OPT==2){
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width1 + K1 - 1, tile_width1 + K1 - 1, 1);
		conv21_forward_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, B, M, C, H, W, K);
	}
	else if(C==C2 && USE_OPT==2){
		w_grid = ceil(w_out/(1.0 * tile_width2));
		h_grid = ceil(h_out/(1.0 * tile_width2));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width2 + K2 - 1, tile_width2 + K2 - 1, 1);
		conv22_forward_kernel <<<dimGrid, dimBlock>>> (device_y, device_x, B, M, C, H, W, K);
	}
	else if(C==C1 && USE_OPT==3){
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width1, tile_width1, 1);
		conv31_forward_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, device_k, B, M, C, H, W, K);
	}
	else if(C==C2 && USE_OPT==3){
		w_grid = ceil(w_out/(1.0 * tile_width2));
		h_grid = ceil(h_out/(1.0 * tile_width2));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width2, tile_width2, 1);
		conv32_forward_kernel <<<dimGrid, dimBlock>>> (device_y, device_x, device_k, B, M, C, H, W, K);
	}
	else if(USE_OPT==4){
		half_t *device_y_half,*device_x_half,*device_k_half;
		int size;
		cudaMalloc((void**)(&device_x_half), (C * H * W * B) * sizeof(half_t));
		cudaMalloc((void**)(&device_y_half), (M * h_out * w_out * B) * sizeof(half_t));
		cudaMalloc((void**)(&device_k_half), (C * K * K * M) * sizeof(half_t));

		size = (C * H * W * B);
		Float2HalfKernel <<<(size-1)/256+1,256>>> (device_x_half,device_x,size);
		cudaDeviceSynchronize();

		size = (C * K * K * M);
		Float2HalfKernel <<<(size-1)/256+1,256>>> (device_k_half,device_k,size);
		cudaDeviceSynchronize();

		w_grid = ceil(w_out/(1.0 * tile_width2));
		h_grid = ceil(h_out/(1.0 * tile_width2));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width2, tile_width2, 1);
		conv4_forward_kernel <<<dimGrid, dimBlock>>> (device_y_half, device_x_half, device_k_half, B, M, C, H, W, K);
		cudaDeviceSynchronize();

		size = (M * h_out * w_out * B);
		Half2FloatKernel <<<(size-1)/256+1,256>>> (device_y,device_y_half,size);
		
	}
	else if(C==C1 && USE_OPT==5){
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width1+ K1 - 1, tile_width1+ K1 - 1, 1);
		conv51_forward_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, B, M, C, H, W, K);
	}
	else if(C==C2 && USE_OPT==5){
		w_grid = ceil(w_out/(1.0 * tile_width2));
		h_grid = ceil(h_out/(1.0 * tile_width2));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width2+ K2 - 1, tile_width2+ K2 - 1, 1);
		conv52_forward_kernel <<<dimGrid, dimBlock>>> (device_y, device_x, B, M, C, H, W, K);
	}
	else if(C==C1 && USE_OPT==6){
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width1, tile_width1, 1);
		conv31_forward_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, device_k, B, M, C, H, W, K);
	}
	else if(C==C2 && USE_OPT==6){
		half_t *device_y_half,*device_x_half,*device_k_half;
		int size;
		cudaMalloc((void**)(&device_x_half), (C * H * W * B) * sizeof(half_t));
		cudaMalloc((void**)(&device_y_half), (M * h_out * w_out * B) * sizeof(half_t));
		cudaMalloc((void**)(&device_k_half), (C * K * K * M) * sizeof(half_t));

		size = (C * H * W * B);
		Float2HalfKernel <<<(size-1)/256+1,256>>> (device_x_half,device_x,size);
		cudaDeviceSynchronize();

		size = (C * K * K * M);
		Float2HalfKernel <<<(size-1)/256+1,256>>> (device_k_half,device_k,size);
		cudaDeviceSynchronize();

		w_grid = ceil(w_out/(1.0 * tile_width2));
		h_grid = ceil(h_out/(1.0 * tile_width2));
		Y = h_grid*w_grid;
		dimGrid = dim3(M,Y,B);
		dimBlock = dim3(tile_width2, tile_width2, 1);
		conv4_forward_kernel <<<dimGrid, dimBlock>>> (device_y_half, device_x_half, device_k_half, B, M, C, H, W, K);
		cudaDeviceSynchronize();

		size = (M * h_out * w_out * B);
		Half2FloatKernel <<<(size-1)/256+1,256>>> (device_y,device_y_half,size);
	}
	else if(USE_OPT==7){
		// init
		float* device_X_unroll;
		dim3 dimGrid1,dimGrid2;
		dim3 dimBlock1,dimBlock2;

		printf("memory allocated for one section: %lld bytes\n", C*K*K*h_out*w_out*(B/SEC)*sizeof(float));
		cudaMalloc((void**)(&device_X_unroll), (C*K*K*h_out*w_out*(B/SEC))*sizeof(float));
		// unroll kernel 
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid1 = dim3(C,Y,B/SEC);
		dimBlock1 = dim3(tile_width1, tile_width1, 1);

		// mat mul kernel
		int num_W_Unroll_Rows = M;
		int num_W_Unroll_Columns = K*K*C;
		int num_X_Unroll_Rows = K*K*C;
		int num_X_Unroll_Columns = w_out*h_out;
		int num_Y_Unroll_Rows = M;
		int num_Y_Unroll_Columns = w_out*h_out;

		dimGrid2 = dim3(ceil((1.0*num_Y_Unroll_Columns)/TILE_SIZE),ceil((1.0*num_Y_Unroll_Rows)/TILE_SIZE),B/SEC);
		dimBlock2 = dim3(TILE_SIZE, TILE_SIZE, 1);
		
		printf("num_W_Unroll_Rows x num_W_Unroll_Columns : %d x %d \n",num_W_Unroll_Rows,num_W_Unroll_Columns);
		printf("num_X_Unroll_Rows x num_X_Unroll_Columns : %d x %d \n",num_X_Unroll_Rows,num_X_Unroll_Columns);
		printf("num_Y_Unroll_Rows x num_Y_Unroll_Columns : %d x %d \n",num_Y_Unroll_Rows,num_Y_Unroll_Columns);
		
		int device_xstart,device_ystart;
		for(int sec=0;sec<SEC;sec++){
			device_xstart = sec*B/SEC * C * H * W;
			device_ystart = sec*B/SEC * M * h_out * w_out;
			unroll_kernel<<<dimGrid1,dimBlock1>>> (device_X_unroll, &device_x[device_xstart], B/SEC, C, H, W, K);
			cudaDeviceSynchronize();
			//printf("Unroll Success!\n");
			
			conv_forward_matmul_kernel<<<dimGrid2,dimBlock2>>>(device_k, device_X_unroll, &device_y[device_ystart], num_W_Unroll_Rows, num_W_Unroll_Columns, num_X_Unroll_Rows,num_X_Unroll_Columns,num_Y_Unroll_Rows,num_Y_Unroll_Columns);
			//printf("Shared MatMul Success!\n");
			cudaDeviceSynchronize();
		}
	}
	else if(C==C1 && USE_OPT==8){
		float* device_X_unroll;
		dim3 dimGrid1,dimGrid2;
		dim3 dimBlock1,dimBlock2;

		printf("memory allocated for one section: %lld bytes\n", C*K*K*h_out*w_out*(B/SEC)*sizeof(float));
		cudaMalloc((void**)(&device_X_unroll), (C*K*K*h_out*w_out*(B/SEC))*sizeof(float));
		// unroll kernel 
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid1 = dim3(C,Y,B/SEC);
		dimBlock1 = dim3(tile_width1, tile_width1, 1);

		// mat mul kernel
		int num_W_Unroll_Rows = ((M-1)/TILE_SIZE+1)*TILE_SIZE;
		int num_W_Unroll_Columns = ((K*K*C-1)/TILE_SIZE+1)*TILE_SIZE;
		int num_X_Unroll_Rows = K*K*C;
		int num_X_Unroll_Columns = w_out*h_out;
		int num_Y_Unroll_Rows = M;
		int num_Y_Unroll_Columns = w_out*h_out;

		dimGrid2 = dim3(ceil((1.0*num_Y_Unroll_Columns)/TILE_SIZE),ceil((1.0*num_Y_Unroll_Rows)/TILE_SIZE),B/SEC);
		dimBlock2 = dim3(TILE_SIZE, TILE_SIZE, 1);
		
		printf("num_W_Unroll_Rows x num_W_Unroll_Columns : %d x %d \n",num_W_Unroll_Rows,num_W_Unroll_Columns);
		printf("num_X_Unroll_Rows x num_X_Unroll_Columns : %d x %d \n",num_X_Unroll_Rows,num_X_Unroll_Columns);
		printf("num_Y_Unroll_Rows x num_Y_Unroll_Columns : %d x %d \n",num_Y_Unroll_Rows,num_Y_Unroll_Columns);
		
		int device_xstart,device_ystart;
		for(int sec=0;sec<SEC;sec++){
			device_xstart = sec*B/SEC * C * H * W;
			device_ystart = sec*B/SEC * M * h_out * w_out;
			unroll_kernel<<<dimGrid1,dimBlock1>>> (device_X_unroll, &device_x[device_xstart], B/SEC, C, H, W, K);
			cudaDeviceSynchronize();
			//printf("Unroll Success!\n");
			
			conv_forward_matmul_constmemory_kernel1<<<dimGrid2,dimBlock2>>>(device_X_unroll, &device_y[device_ystart], num_W_Unroll_Rows, num_W_Unroll_Columns, num_X_Unroll_Rows,num_X_Unroll_Columns,num_Y_Unroll_Rows,num_Y_Unroll_Columns);
			//printf("Shared MatMul Success!\n");
			cudaDeviceSynchronize();
		}
	}
	else if(C==C2 && USE_OPT==8){
		float* device_X_unroll;
		dim3 dimGrid1,dimGrid2;
		dim3 dimBlock1,dimBlock2;

		printf("memory allocated for one section: %lld bytes\n", C*K*K*h_out*w_out*(B/SEC)*sizeof(float));
		cudaMalloc((void**)(&device_X_unroll), (C*K*K*h_out*w_out*(B/SEC))*sizeof(float));
		// unroll kernel 
		w_grid = ceil(w_out/(1.0 * tile_width1));
		h_grid = ceil(h_out/(1.0 * tile_width1));
		Y = h_grid*w_grid;
		dimGrid1 = dim3(C,Y,B/SEC);
		dimBlock1 = dim3(tile_width1, tile_width1, 1);

		// mat mul kernel
		int num_W_Unroll_Rows = ((M-1)/TILE_SIZE+1)*TILE_SIZE;
		int num_W_Unroll_Columns = ((K*K*C-1)/TILE_SIZE+1)*TILE_SIZE;
		int num_X_Unroll_Rows = K*K*C;
		int num_X_Unroll_Columns = w_out*h_out;
		int num_Y_Unroll_Rows = M;
		int num_Y_Unroll_Columns = w_out*h_out;

		dimGrid2 = dim3(ceil((1.0*num_Y_Unroll_Columns)/TILE_SIZE),ceil((1.0*num_Y_Unroll_Rows)/TILE_SIZE),B/SEC);
		dimBlock2 = dim3(TILE_SIZE, TILE_SIZE, 1);
		
		printf("num_W_Unroll_Rows x num_W_Unroll_Columns : %d x %d \n",num_W_Unroll_Rows,num_W_Unroll_Columns);
		printf("num_X_Unroll_Rows x num_X_Unroll_Columns : %d x %d \n",num_X_Unroll_Rows,num_X_Unroll_Columns);
		printf("num_Y_Unroll_Rows x num_Y_Unroll_Columns : %d x %d \n",num_Y_Unroll_Rows,num_Y_Unroll_Columns);
		
		int device_xstart,device_ystart;
		for(int sec=0;sec<SEC;sec++){
			device_xstart = sec*B/SEC * C * H * W;
			device_ystart = sec*B/SEC * M * h_out * w_out;
			unroll_kernel<<<dimGrid1,dimBlock1>>> (device_X_unroll, &device_x[device_xstart], B/SEC, C, H, W, K);
			cudaDeviceSynchronize();
			//printf("Unroll Success!\n");
			
			conv_forward_matmul_constmemory_kernel2<<<dimGrid2,dimBlock2>>>(device_X_unroll, &device_y[device_ystart], num_W_Unroll_Rows, num_W_Unroll_Columns, num_X_Unroll_Rows,num_X_Unroll_Columns,num_Y_Unroll_Rows,num_Y_Unroll_Columns);
			//printf("Shared MatMul Success!\n");
			cudaDeviceSynchronize();
		}
	}
	else if(USE_OPT==9){
		int num_Y_Unroll_Rows = M;
		int num_Y_Unroll_Columns = w_out*h_out;
		dimGrid = dim3(ceil((1.0*num_Y_Unroll_Columns)/TILE_SIZE),ceil((1.0*num_Y_Unroll_Rows)/TILE_SIZE),B);
		dimBlock = dim3(TILE_SIZE, TILE_SIZE, 1);
		conv_forward_matmul_builtin_unroll_kernel <<<dimGrid, dimBlock >>> (device_y, device_x, device_k, B, M, C, H, W, K);
		printf("num_Y_Unroll_Rows x num_Y_Unroll_Columns : %d x %d \n",num_Y_Unroll_Rows,num_Y_Unroll_Columns);
	}

	if(USE_OPT!=7 && USE_OPT!=8) cudaDeviceSynchronize();

	if(USE_OPT!=4 && USE_OPT!=9){
		printf("B,M,C,H,W,K is : %d,%d,%d,%d,%d,%d \n",B,M,C,H,W,K);
		printf("h_out,w_out,w_grid,h_grid,Y is : %d,%d,%d,%d,%d \n",h_out,w_out,w_grid,h_grid,Y);
	}
	
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
	int h_out = H-K+1;
	int w_out = W-K+1;
	//host_y = (float *)malloc((M * h_out * w_out * B)*sizeof(float));
	cudaMemcpy(host_y, device_y, (M * h_out * w_out* B) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
	cudaFree(device_x);
    cudaFree(device_y);
	cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
