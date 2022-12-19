//* Host c++ *------------------------------------------
//______________________________________________________        
//  begin:  Fri. Oct. 07, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//______________________________________________________
//------------------------------------------------------

#ifndef GG_DEC_H
#define GG_DEC_H

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

// the two value can be set into class of NET, but in order to wirte more 
// function in "gg_def.cpp" file, we set the two values here
const double PI = 3.141592653;
const int Nx = 100, Ny = 100;

class NET {
public:
    int num_grain;
    double rad_ini;                                                          // each grain size
    double rad_dif;                                                             // flag for seting the distance between each grain
    NET(int, double, double);
};

class PPF {
public:
    double deltaT;                                                           // also we can use the extern to declare the variable, then define in the "gg_def" file
    double garma;
    double deltaE;
    double deltaX;
    double K;
    double Qb;
    double B;
    double R;
    double T;
    double thigma;
    double W; 
    double a;
    double Mij;
    double M;
public:
    PPF();                                                                   // dierct define all private variable in "gg_def.cpp" file
};

struct GRAIN {
    int idg;
    double phi;
};

struct OUTPUT {
    int step;
    int stp_vtk;
    int stp_tra;
};

// other function
bool LessSort(GRAIN, GRAIN); 
void sys_time(char buf_time[]);
void output_info(NET, PPF, OUTPUT, char globalName[]);
void init_grain(NET, char globalFile[], std::vector<struct GRAIN> phi[][Ny], 
                std::vector<struct GRAIN> phi_n[][Ny]);
void output_grid(char vtkName[], char traName[], 
                 int, int, bool, bool, PPF, 
                 std::vector<struct GRAIN> phi_n[][Ny], 
                 std::vector<struct GRAIN>::iterator it_pn);

// boundary condition
void periodic(int, int, int pd[][2]);
void ar_points(int, int, int &, int &, int &, int&);
double find_phi(int, int, int, std::vector<struct GRAIN> phi[][Ny]);

#endif
