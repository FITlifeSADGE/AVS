/**
 * @file BatchMandelCalculator.cc
 * @author Lukáš Kaprál <xkapra00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include <immintrin.h>
#include <cstring>

#include "BatchMandelCalculator.h"

#define blockSize 96

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
	real = (float *)aligned_alloc(64, width * 2 * sizeof(float));
	imag = (float *)aligned_alloc(64, height * 2 * sizeof(float));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	free(real);
	free(imag);
}


int * BatchMandelCalculator::calculateMandelbrot () {
	float *preal = real;
	float *pimag = imag;

	int *pdata_tmp = new int[width * height]();

	int *tmp_arr = tmp;

	float x[width];
	int counter = blockSize;

	//size_t blockSize = 64;

	std::fill(pdata_tmp, pdata_tmp + width * height, limit);

	for (int i = 0; i < width; i++) {
		x[i] = x_start + i * dx;
	}
	for (int i = 0; i < height / 2; i++) {
		counter = blockSize;
		size_t index = i * width;
		size_t mirror_index = (height - i - 1) * width;
		const float y = (float) y_start + i * (float) dy; // current imaginary value
		counter = blockSize;
		for (int j = 0; j < width/blockSize; j++) {
			counter = blockSize;
			for (int k = 0; k < limit; ++k) {
				if (!counter) {
					break;
				}
				counter = blockSize;
				#pragma omp simd
				for (size_t block = 0; block < blockSize; block++) {
					const size_t jGlobal = j * blockSize + block;
					if (!k) {
						preal[jGlobal] = x[jGlobal];
						pimag[jGlobal] = y;
					}
					float r2 = preal[jGlobal] * preal[jGlobal];
					float i2 = pimag[jGlobal] * pimag[jGlobal];

					int komar = r2 + i2 > 4.0f;
					pdata_tmp[jGlobal + index] = komar ? k : pdata_tmp[jGlobal + index];
					pdata_tmp[jGlobal + mirror_index] = komar ? k : pdata_tmp[jGlobal + mirror_index];

					counter -= komar;

					pimag[jGlobal] = 2.0f * preal[jGlobal] * pimag[jGlobal] + y;
					preal[jGlobal] = r2 - i2 + x[jGlobal];
				}
			}
		}	
	}
    return pdata_tmp;
}

