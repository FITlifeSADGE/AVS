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
	data = (int *)_mm_malloc(height * width * sizeof(int), 32);
	if (data == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
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
	_mm_free(data);
	data = NULL;
	_mm_free(real);
	real = NULL;
	_mm_free(imag);
	imag = NULL;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	// @TODO implement Mandelbrot calculation using SIMD
	int *pdata = data;
	float *preal = real;
	float *pimag = imag;

	float r2;
	float i2;
	int counter = 0;

	float x;
	float y;

	for (int i = 0; i < height * width; i++) {
		pdata[i] = limit;
	}
    for (int i = 0; i < height/2; i++)
    {
		int index = i * width;
		int mirror_index = (height - i - 1) * width;
		y = (float) y_start + i * (float) dy; // current imaginary value
		counter = 0;
        for (int k = 0; ((k < limit) && (counter < width)); ++k)
        {
			#pragma omp simd
			for (int j = 0; j < width; j++) {
				x = (float) x_start + j * (float) dx; // current real value
				
				if (k == 0) {
					preal[j] = x;
					pimag[j] = y;
				}

				r2 = preal[j] * preal[j];
				i2 = pimag[j] * pimag[j];
				//#pragma omp simd reduction(+:counter)
				//if ((r2 + i2 > 4.0f) && ((*(pdata + j)) == limit)) {
					//(*(pdata + j)) = k;
				if ((r2 + i2 > 4.0f) && (pdata[j + index] == limit)) {
					pdata[j + index] = k;
					pdata[mirror_index + j] = k;
					counter++;
				}
				else {
					pimag[j] = 2.0f * preal[j] * pimag[j] + y;
					preal[j] = r2 - i2 + x;
				}
			}
        }
		//pdata += width;
    }
    return data;
}
