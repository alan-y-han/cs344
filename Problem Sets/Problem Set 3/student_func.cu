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

#include <cmath>
#include "utils.h"

typedef float(*reduceFn_t)(float, float);

const unsigned int BLOCK_1D = 1024;

__device__
float calcMin(float a, float b)
{
    return min(a, b);
}

__device__
float calcMax(float a, float b)
{
    return max(a, b);
}

__device__ reduceFn_t dp_calcMin = calcMin;
__device__ reduceFn_t dp_calcMax = calcMax;

reduceFn_t hp_calcMin;
reduceFn_t hp_calcMax;

__global__
void reduce(float* const values, const unsigned int noOfElems, reduceFn_t reduceFn)
{
    int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalPos >= noOfElems) return;

    int localPos = threadIdx.x;

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (localPos < s && ((globalPos + s) < noOfElems))
        {
            values[globalPos] = reduceFn(values[globalPos], values[globalPos + s]);
        }
        __syncthreads();
    }
}

__global__
void gather(float* const values, const unsigned int noOfElems)
{
    int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalPos >= noOfElems) return;
    values[globalPos] = values[globalPos * blockDim.x];
}

__global__
void atomic_histo(unsigned int* d_bins, const float* d_in, const int BIN_COUNT, const unsigned int noOfElems, const float lumMin, const float lumRange)
{
    int globalPos = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalPos >= noOfElems) return;
    float itemToSort = d_in[globalPos];
    int myBin = (itemToSort - lumMin) / lumRange * BIN_COUNT;
    atomicAdd(&(d_bins[myBin]), 1);
}

__global__
void globalAddReduce(unsigned int *const values, const unsigned int noOfElems, const int shift)
{
    int globalPos = blockIdx.x * blockDim.x + threadIdx.x;

    int posInArray = noOfElems - 1 - (globalPos * (shift << 1));
    int posOfReduceElem = posInArray - shift;

    if (posInArray < 0 || posOfReduceElem < 0) return;

    values[posInArray] += values[posOfReduceElem];
}

__global__
void globalAddDownsweep(unsigned int *const values, const unsigned int noOfElems, const int shift)
{
    int globalPos = blockIdx.x * blockDim.x + threadIdx.x;

//    int posInArray = noOfElems - 1 -
}

float cudaReduce(unsigned int noOfElems, const float *const d_array, reduceFn_t d_reduceFn)
{
    float* d_arrayCopy;
    checkCudaErrors(cudaMalloc(&d_arrayCopy, sizeof(float) * noOfElems));
    checkCudaErrors(cudaMemcpy(d_arrayCopy, d_array, sizeof(float) * noOfElems, cudaMemcpyDeviceToDevice));

    unsigned int elemsToReduce = noOfElems;
    unsigned int grid_1D = (elemsToReduce + BLOCK_1D - 1) / BLOCK_1D;
    do
    {
        reduce<<<grid_1D, BLOCK_1D>>>(d_arrayCopy, elemsToReduce, d_reduceFn);
        checkCudaErrors(cudaGetLastError());

        elemsToReduce = grid_1D;
        grid_1D = (elemsToReduce + BLOCK_1D - 1) / BLOCK_1D;

        gather<<<grid_1D, BLOCK_1D>>>(d_arrayCopy, elemsToReduce);
        checkCudaErrors(cudaGetLastError());
    } while (elemsToReduce > 1);

    float result;
    checkCudaErrors(cudaMemcpy(&result, d_arrayCopy, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_arrayCopy));
    return result;
}

void cudaCDF(const float *const d_inputHisto, unsigned int *const d_outputBins, const unsigned int noOfElems)
{
    cudaMemcpy(d_outputBins, d_inputHisto, sizeof(float) * noOfElems, cudaMemcpyDeviceToDevice);

    unsigned int noElemsNearestPow2 = round(pow(2, ceil(log2(static_cast<float>(noOfElems))))); // compute the next highest power of 2 of 32-bit

    // Reduce step
    for (int shift = 1, noThreads = (noElemsNearestPow2 >> 1);
         shift < noElemsNearestPow2;
         shift <<= 1, noThreads >>= 1)
    {
        unsigned int grid_1D = (noThreads + BLOCK_1D - 1) / BLOCK_1D;
        globalAddReduce<<<grid_1D, BLOCK_1D>>>(d_outputBins, noOfElems, shift);
        checkCudaErrors(cudaGetLastError());
    }

    // Downsweep step
    int identityValue = 0;
    cudaMemcpy(&d_outputBins[noOfElems - 1], &identityValue, sizeof(float), cudaMemcpyHostToDevice);
    for (int shift = (noElemsNearestPow2 >> 1), noThreads = 1;
         shift > 0;
         shift >>= 1, noThreads <<= 1)
    {
        unsigned int grid_1D = (noThreads + BLOCK_1D - 1) / BLOCK_1D;
        globalAddDownsweep<<<grid_1D, BLOCK_1D>>>(d_outputBins, noOfElems, shift);
        checkCudaErrors(cudaGetLastError());
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
    /*Here are the steps you need to implement
      1) find the minimum and maximum value in the input logLuminance channel
         store in min_logLum and max_logLum
      2) subtract them to find the range
      3) generate a histogram of all the values in the logLuminance channel using
         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
      4) Perform an exclusive scan (prefix sum) on the histogram to get
         the cumulative distribution of luminance values (this should go in the
         incoming d_cdf pointer which already has been allocated for you)       */

    const unsigned int noOfElems = numRows * numCols;

    cudaMemcpyFromSymbol(&hp_calcMin, dp_calcMin, sizeof(reduceFn_t));
    cudaMemcpyFromSymbol(&hp_calcMax, dp_calcMax, sizeof(reduceFn_t));

    // 1.
    min_logLum = cudaReduce(noOfElems, d_logLuminance, hp_calcMin);
    max_logLum = cudaReduce(noOfElems, d_logLuminance, hp_calcMax);
    std::cout << "GPU min: " << min_logLum << std::endl;
    std::cout << "GPU max: " << max_logLum << std::endl;

    // 2.
    float logLumRange = max_logLum - min_logLum;

    // 3.
    unsigned int grid_1D = (noOfElems + BLOCK_1D - 1) / BLOCK_1D;

    atomic_histo<<<grid_1D, BLOCK_1D>>>(d_cdf, d_logLuminance, numBins, noOfElems, min_logLum, logLumRange);
    checkCudaErrors(cudaGetLastError());

    // 4.
//    cudaCDF(d_logLuminance, d_cdf, noOfElems);

}
