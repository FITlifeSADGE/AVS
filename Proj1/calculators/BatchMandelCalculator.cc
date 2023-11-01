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

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
	real = (float *)_mm_malloc(width * 2 * sizeof(float), 64);
	if (real == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
	imag = (float *)_mm_malloc(height * 2 * sizeof(float), 64);
	if (imag == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(real);
	real = NULL;
	_mm_free(imag);
	imag = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
	float *preal = real;
	float *pimag = imag;

	int *pdata_tmp = new int[width * height]();
	int counter = 0;

	float r2;
	float i2;

	float x[width];
	float y;


	size_t blockSize = 96;

	std::fill(pdata_tmp, pdata_tmp + width * height, limit);

	for (int i = 0; i < width; i++) {
		x[i] = x_start + i * dx;
	}

	for (int i = 0; i < (height / 2); i++) {
		counter = 0;
		int index = i * width;
		int mirror_index = (height - i - 1) * width;
		y = (float) y_start + i * (float) dy; // current imaginary value
		for (int k = 0; (k < limit) && (counter < width); ++k) {
			for (size_t j = 0; j < width/blockSize; j++) {
				#pragma omp simd
				for (int block = 0; block < blockSize; block++) {
					const size_t jGlobal = j * blockSize + block;
					if (!k) {
						preal[jGlobal] = x[jGlobal];
						pimag[jGlobal] = y;
					}
					r2 = preal[jGlobal] * preal[jGlobal];
					i2 = pimag[jGlobal] * pimag[jGlobal];

					((r2 + i2 > 4.0f) && (pdata_tmp[jGlobal + index] == limit)) ? (pdata_tmp[jGlobal + index] = k), (pdata_tmp[mirror_index + jGlobal] = k), counter++ : 0;

					pimag[jGlobal] = 2.0f * preal[jGlobal] * pimag[jGlobal] + y;
					preal[jGlobal] = r2 - i2 + x[jGlobal];
				}

			}
		}
	}
    return pdata_tmp;
}
