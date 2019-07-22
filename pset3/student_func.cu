/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>

// BLOCKSIZE has to be of 2^s
#define BLOCKSIZE 1024;

// Kernel for finding min and max
__global__
void block_min_max(const float* const d_in,
		  float* d_out,
		  size_t size,
		  bool wantMin)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Declear shared data
  extern __shared__ float shdata[];
  
  // Because we assume every block has a size of 2^s suitable for binary reduction
  // We need to fill the holes in the last block that are out of images size
  if (idx < size) {
    // Load data to shared memory
    shdata[threadIdx.x] = d_in[idx];
  } else {
    // File holes in the last block
    if (wantMin) {
      shdata[threadIdx.x] = FLT_MAX;
    } else {
      shdata[threadIdx.x] = FLT_MIN;
    }
  }
  __syncthreads();

  // Reduction in global memory
  for (size_t s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (wantMin) {
        shdata[threadIdx.x] = shdata[threadIdx.x] < shdata[threadIdx.x+s] ?
	        shdata[threadIdx.x] : shdata[threadIdx.x+s];
      } else {
        shdata[threadIdx.x] = shdata[threadIdx.x] > shdata[threadIdx.x+s] ?
	        shdata[threadIdx.x] : shdata[threadIdx.x+s];
      }

      __syncthreads();
    }
  }

  // Thread 0 writes result for this block back to out
  if (threadIdx.x == 0) {
    d_out[blockIdx.x] = shdata[0];
  }
}

void min_max_reduce(const float* const d_in,
		    float &minmax,
		    size_t size,
		    bool wantMin) {
  int numThreads = BLOCKSIZE;
  int numBlocks = size / numThreads + 1;

  const float* d_temp_in;
  float d_temp_out[numBlocks];
  size_t size_in = size;
  d_temp_in = d_in;
  while (true) {
    block_min_max<<<numBlocks, numThreads, numThreads * sizeof(float)>>>
	    (d_temp_in, d_temp_out, size_in, wantMin);

    // Found the result and return
    if (numBlocks == 1) {
        minmax = d_temp_out[0];
	return;
    }
    size_in = numBlocks;
    numBlocks = numBlocks / numThreads + 1;
    d_temp_in = d_temp_out;
  }
}


__global__ 
void gen_hist(const float * const d_logLuminance,
		    int * d_bins,
		    const float min_logLum,
		    const float lumRange,
		    const size_t numBins,
		    size_t size,
		    bool useLocalBin) {
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
   
   if (idx >= size) return;

   // Use atomicAdd
   if (!useLocalBin) {
      float lum_value = d_logLuminance[idx];
      size_t bin = (lum_value - min_logLum) / lumRange * numBins;
      atomicAdd(&(d_bins[bin]), 1);
   }
}



// This function implements the Hillis Steele algorithm
// This version is from the NVIDIA GPU Gems.
/* This version can handle arrays only as large as can be
   processed by a single thread block running on one multiprocessor
   of a GPU. */
__global__ 
void prescan(const int* const d_bins, 
		unsigned int* const d_cdf,
	       	const size_t numBins) {
  extern __shared__ float shdata[]; // Shared memory allocated on kernel launch
  int pout = 0, pin = 1;
  int thid = threadIdx.x;

  // Load input into shared memory
  // This is a exclusive scan, so shift right by one
  // and set first element to 0.
  shdata[pout * numBins + thid] = (thid > 0) ? d_bins[thid - 1] : 0;
  __syncthreads();

  for (int offset = 1; offset < numBins; offset *= 2) {
    // Swap double buffer indices
    pout = 1 - pout; 
    pin = 1 - pout;
   
    // Double buffered 
    if (thid >= offset) {
      shdata[pout * numBins + thid] += shdata[pin * numBins + thid - offset];
    } else {
      shdata[pout * numBins + thid] = shdata[pin * numBins + thid];
    }
    __syncthreads();
  }
  d_cdf[thid] = shdata[pout * numBins + thid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    min_max_reduce(d_logLuminance, min_logLum, numRows*numCols, true);
    min_max_reduce(d_logLuminance, max_logLum, numRows*numCols, false);

    // Allocate memory to store bins
    int *d_bins;
    checkCudaErrors(cudaMalloc(&d_bins, numBins * sizeof(size_t)));

    // Launch kernel to generate histogram
    const int numThreads = BLOCKSIZE;
    const int numBlocks = numRows*numCols / numThreads + 1;
    gen_hist<<<numBlocks, numThreads>>>(d_logLuminance, d_bins, 
		    min_logLum, max_logLum - min_logLum, 
		    numBins, numRows*numCols, false);
    
    // numBins is defined to be 1024 in another file
    prescan<<<1, numBins, 2 * numBins * sizeof(unsigned)>>>(d_bins, d_cdf, numBins);

    checkCudaErrors(cudaFree(d_bins));
}
