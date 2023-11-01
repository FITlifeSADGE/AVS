/**
 * @file LineMandelCalculator.cc
 * @author Lukas Kapral (xkapra00@fit.vutbr.cz)
 * @brief Implementation of Mandelbrot calculator that uses SIMD palalelization over Lines
 * @date 2023-10-16
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <cstring>

#include "LineMandelCalculator.h"

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory
	real = (float *)_mm_malloc(width * 2 * sizeof(float), 64);
	if (real == nullptr) {
		std::cerr << "Error: Failed to allocate memory for real" << std::endl;
		exit(1);
	}
	imag = (float *)_mm_malloc(height * 2 * sizeof(float), 64);
	if (imag == nullptr) {
		std::cerr << "Error: Failed to allocate memory for imag" << std::endl;
		exit(1);
	}
}

LineMandelCalculator::~LineMandelCalculator()
{
	// @TODO cleanup the memory
	_mm_free(real);
	real = NULL;
	_mm_free(imag);
	imag = NULL;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	// @TODO implement Mandelbrot calculation using SIMD
	float *preal = real;
	float *pimag = imag;

	int *pdata_tmp = new int[width * height]();
	int counter = 0;

	float r2;
	float i2;

	float x;
	float y;

	std::fill(pdata_tmp, pdata_tmp + width * height, limit);

    for (int i = 0; i < height/2; i++)
    {
		int index = i * width;
		int mirror_index = (height - i - 1) * width;
		y = (float) y_start + i * (float) dy; // current imaginary value
		counter = 0;
        for (int k = 0; (k < limit) && (counter < width); ++k)
        {
			#pragma omp simd
			for (int j = 0; j < width; j++) {
				x = (float) x_start + j * (float) dx; // current real value
				
				if (!k) {
					preal[j] = x;
					pimag[j] = y;
				}

				r2 = preal[j] * preal[j];
				i2 = pimag[j] * pimag[j];

				((r2 + i2 > 4.0f) && (pdata_tmp[j + index] == limit)) ? (pdata_tmp[j + index] = k), (pdata_tmp[mirror_index + j] = k), counter++ : 0;

				pimag[j] = 2.0f * preal[j] * pimag[j] + y;
				preal[j] = r2 - i2 + x;
			}
        }
    }
    return pdata_tmp;
}
