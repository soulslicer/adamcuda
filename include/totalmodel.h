#ifndef TOTALMODEL
#define TOTALMODEL
#include <iostream>
#include <vector>

struct TotalModel
{
    static const int NUM_FAST_VERTICES = 180;
    static const int NUM_VERTICES = 18540;
    static const int NUM_FACES = 36946;// 111462;

    static const int NUM_JOINTS = 62;		//(SMPL-hands)22 + lhand20 + rhand20
    static const int NUM_POSE_PARAMETERS = NUM_JOINTS * 3;
    static const int NUM_SHAPE_COEFFICIENTS = 30;
    static const int NUM_EXP_BASIS_COEFFICIENTS = 200; //Facial expression
};

#endif
