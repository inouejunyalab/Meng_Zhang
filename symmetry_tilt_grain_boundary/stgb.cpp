#include <stdio.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <fstream>
#include <vector>
#include <omp.h>
#include <assert.h>
#include <cctype>
#include <stdlib.h>
#include "stgb_b.h"

int main() {
    std::vector<struct COORD> coord;
    std::vector<struct COORD>::iterator itc;

    double lattice = 2.8553;
    double unit0_xyz[3][3] = { 0.0 };
    double matrix0_xyz[3][3] = { -1, 1,-2, 1, -1, -1,  1, 1, 0 };    // the vector along z-axis should be calcualted acoording to your need
    double length_box[3] = { 34.97014031, 49.45524671, 32.30403188 };
    char file_name[100] = "./fe.dat";						        // writing the data to file

    // set the unit of the box vector
    for (int i = 0; i < 3; i++) {
        double sum0 = sqrt(pow(matrix0_xyz[i][0], 2) + pow(matrix0_xyz[i][1], 2) + pow(matrix0_xyz[i][2], 2));
        for (int j = 0; j < 3; j++) {
            unit0_xyz[i][j] = matrix0_xyz[i][j] / sum0;
        }
    }

    //-----creating the grain 1
    CRY_BOX matrix0(unit0_xyz, length_box, lattice);
    matrix0.get_euler_angle();										// get the euler matrix for grain 1
    build_crystal(matrix0, coord, itc, 1);						    // creating grain 1
    symm_crystal(matrix0, coord, itc);
    length_box[0] *= 2;

    write_crystal(file_name, length_box, 2, coord, itc);
    for (int i = 0; i < 3; i++) {
        std::cout << matrix0.xyz_vect[i][0] << " " << matrix0.xyz_vect[i][1] << " " << matrix0.xyz_vect[i][2] << " " << std::endl;
    }

    return 0;
}