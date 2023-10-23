/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
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
	data = (int *)_mm_malloc(height * width * sizeof(int), 32);
	if (data == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
	real = (float *)_mm_malloc(width * 2 * sizeof(float), 64);
	if (real == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
	imag = (float *)_mm_malloc(height * 2 * sizeof(float), 64);
	if (pre_real == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
	pre_real = (float *)_mm_malloc(width * 2 * sizeof(float), 64);
	if (real == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
	pre_imag = (float *)_mm_malloc(height * 2 * sizeof(float), 64);
	if (pre_imag == nullptr) {
		std::cerr << "Error: Failed to allocate memory for data" << std::endl;
		exit(1);
	}
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
	_mm_free(real);
	real = NULL;
	_mm_free(imag);
	imag = NULL;
	_mm_free(pre_real);
	pre_real = NULL;
	_mm_free(pre_imag);
	pre_imag = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
	int *pdata = data;
	float *preal = real;
	float *pimag = imag;
	float *pre_calc_real = pre_real;
	float *pre_calc_imag = pre_imag;


	float r2;
	float i2;
	float zReal;
	float zImag;

	//std::fill(pdata, pdata + height * width, limit);

	for (int i = 0; i < height * width; i++) {
		pdata[i] = limit;
	}

	for (int i = 0; i < height; i++) {
		pre_calc_imag[i] = y_start + i * dy;
	}

	for (int i = 0; i < width; i++) {
		pre_calc_real[i] = x_start + i * dx;
	}
    for (int i = 0; i < height/2; i++)
    {
		//float y = (float) y_start + i * (float) dy; // current imaginary value
        for (int k = 0; k < limit; ++k)
        {
			#pragma omp simd
			for (int j = 0; j < width; j+=4) {
				//float x = (float) x_start + j * (float) dx; // current real value
				if (k == 0) {
					preal[j] = pre_calc_real[j];
					preal[j + 1] = pre_calc_real[j + 1];
					preal[j + 2] = pre_calc_real[j + 2];
					preal[j + 3] = pre_calc_real[j + 3];
					pimag[j] = pre_calc_imag[i];
					pimag[j + 1] = pre_calc_imag[i];
					pimag[j + 2] = pre_calc_imag[i];
					pimag[j + 3] = pre_calc_imag[i];
				}

				r2 = preal[j] * preal[j];
				i2 = pimag[j] * pimag[j];

				if ((r2 + i2 > 4.0f) && (pdata[i * width + j] == limit)) {
					#pragma omp atomic write
					pdata[i * width + j] = k;
					#pragma omp atomic write
					pdata[(height - i - 1) * width + j] = k;
				}
				pimag[j] = 2.0f * preal[j] * pimag[j] + pre_calc_imag[i];
				pimag[j+1] = 2.0f * preal[j+1] * pimag[j+1] + pre_calc_imag[i];
				pimag[j+2] = 2.0f * preal[j+2] * pimag[j+2] + pre_calc_imag[i];
				pimag[j+3] = 2.0f * preal[j+3] * pimag[j+3] + pre_calc_imag[i];
				preal[j] = r2 - i2 + pre_calc_real[j];
				preal[j+1] = r2 - i2 + pre_calc_real[j+1];
				preal[j+2] = r2 - i2 + pre_calc_real[j+2];
				preal[j+3] = r2 - i2 + pre_calc_real[j+3];
			}
        }
    }
    return data;
}