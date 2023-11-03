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
	real = (float *)aligned_alloc(64, width * 2 * sizeof(float));
	imag = (float *)aligned_alloc(64, height * 2 * sizeof(float));
}

LineMandelCalculator::~LineMandelCalculator()
{
	// @TODO cleanup the memory
	free(real);
	real = NULL;
	free(imag);
	imag = NULL;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	// @TODO implement Mandelbrot calculation using SIMD
	float *preal = real;
	float *pimag = imag;

	int *pdata_tmp = new int[width * height]();
	int counter = 0;

	std::fill(pdata_tmp, pdata_tmp + width * height, limit);

    for (int i = 0; i < height/2; i++)
    {
		int index = i * width;
		int mirror_index = (height - i - 1) * width;
		const float y = (float) y_start + i * (float) dy; // current imaginary value
		counter = 0;
        for (int k = 0; (k < limit) && (counter < width); ++k)
        {
			#pragma omp simd reduction(+ : counter) aligned(pdata_tmp, preal, pimag : 64)
			for (int j = 0; j < width; j++) {
				const float x = (float) x_start + j * (float) dx; // current real value
				
				if (!k) {
					preal[j] = x;
					pimag[j] = y;
				}

				float r2 = preal[j] * preal[j];
				float i2 = pimag[j] * pimag[j];

				((r2 + i2 > 4.0f) && (pdata_tmp[j + index] == limit)) ? (pdata_tmp[j + index] = k), (pdata_tmp[mirror_index + j] = k), counter++ : 0;

				pimag[j] = 2.0f * preal[j] * pimag[j] + y;
				preal[j] = r2 - i2 + x;
			}
        }
    }
    return pdata_tmp;
}
