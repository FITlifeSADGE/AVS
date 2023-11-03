/**
 * @file BatchMandelCalculator.h
 * @author Lukáš Kaprál <xkapra00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    // @TODO add all internal parameters
    float *real;
    float *imag;
    int *tmp;

    // float *base_r;
    // float *work_r;
    // float *work_i;
    // int *batch_results;
    // int *results;
};

#endif