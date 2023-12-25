#pragma once

#ifndef LB_STGBB_H
#define LB_STGBB_H

#include <iostream>
#include <vector>
#include <cmath>

const double PI = 3.14159265358979;

class CRY_BOX {
public:
    CRY_BOX(double xyz[][3], double length[3], double la_const);
    CRY_BOX(double xyz[][3]);

    void rot_xyz(double angle[]);
    void get_euler_angle();

    double xyz_vect[3][3];
    double length_box[3];
    double euler_angle[3];
    double euler_matrix[3][3];
    double lattice;
};

struct COORD {                                              // for atoms information
    int id;
    int type;
    double xyz[3];
};

void euler_rotation(CRY_BOX matrix, std::vector<struct COORD>& coord, std::vector<struct COORD>::iterator& itc);
void build_crystal(CRY_BOX, std::vector<struct COORD>&, std::vector<struct COORD>::iterator&, int);
void symm_crystal(CRY_BOX, std::vector<struct COORD>&, std::vector<struct COORD>::iterator&);
void write_crystal(char file_name[], double length_box[], int, std::vector<struct COORD>& coord, std::vector<struct COORD>::iterator& itc);

#endif