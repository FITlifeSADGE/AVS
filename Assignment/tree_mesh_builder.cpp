/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::createCube(const Vec3_t<float> &cubeOffset, unsigned int GridSize, const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;
    if (GridSize <= 1) {
        return buildCube(cubeOffset, field);        
    }
    else {
        float empty_block = field.getIsoLevel() + (sqrt(3) / 2) * (GridSize * mGridResolution);

        Vec3_t<float> midPointNormal((cubeOffset.x + GridSize/2)*mGridResolution, (cubeOffset.y + GridSize/2)*mGridResolution, (cubeOffset.z + GridSize/2)*mGridResolution);

        if (empty_block < evaluateFieldAt(midPointNormal, field)) {
            return 0;
        }
        for (int i = 0; i < 8; i++) {
            #pragma omp task shared(totalTriangles)
            {
                Vec3_t<float> newOffset(cubeOffset.x + (i & 1) * GridSize/2, cubeOffset.y + ((i >> 1) & 1) * GridSize/2, cubeOffset.z + ((i >> 2) & 1) * GridSize/2);
                #pragma omp atomic
                totalTriangles += createCube(newOffset, GridSize/2, field);
            }
        }
    }
    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;
    #pragma omp parallel
    #pragma omp single
    totalTriangles = createCube((0,0,0), mGridSize, field);
    
    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}
