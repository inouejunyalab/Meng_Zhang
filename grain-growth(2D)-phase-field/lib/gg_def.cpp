//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Fri. Oct. 07, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//______________________________________________________
//------------------------------------------------------

#include <fstream>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <io.h>
#include <direct.h>
#include "gg_dec.h"

NET::NET(int ngrain, double rad_i, double rad_d) {
    num_grain = ngrain;
    rad_ini = rad_i;
    rad_dif = rad_d;
}

PPF::PPF() {                                                                    // the parameters are for Cu
    // set the interval time and the whole time
    deltaT = XXXXX;                                                             // timeInterval
    garma = XXXXX;                                                              //  J/m2
    deltaE = XXXXX;
    deltaX = XXXXX;
    K = XXXXX;
    B = XXXXX;                                                                  // Burger's vector
    Qb = XXXXX;                                                                 //  j/mol
    R = XXXXX;                                                                  //  j/(K*mol)
    T = XXXXX;                                                                  //  T is Kelvin's temperature
    thigma = 5 * deltaX;                                                        // delta x is 0.5um£¬use the 'm'
    W = 4 * garma / thigma;        
    a = (2 / PI) * pow(2 * thigma * garma, 0.5);
    Mij = B * 1.1e-13 / K / T * exp(-Qb / (R * T));                             //GB mobility
    M = Mij * PI * PI / (8 * thigma);
}

// sort according to decrease phi value
bool LessSort(GRAIN a, GRAIN b) {
    return (a.phi > b.phi);
}

// obtain the system time
void sys_time(char buf_time[]) {
    time_t time_seconds = time(NULL);
    struct tm now_time;
    localtime_s(&now_time, &time_seconds);
    strftime(buf_time, 64, "%Y-%m-%d:%H:%M:%S", &now_time);
}

// write the grid information and parameters for phase field
void output_info(NET grain, PPF para_pf, OUTPUT out, char globalName[]) {

    std::cout << "==== collect information before evolution ====" << std::endl;
    std::fstream output_info;
    output_info.open(globalName, std::ios::out);
    if (0 != _access_s(globalName, 0)) {                                        // does this file exist?
        std::cout << "We cannot find the global file!!!!" << std::endl;
        exit(-1);
    }
    char buf_time[64];
    sys_time(buf_time);
    output_info << "#creating time: " << buf_time << std::endl;
    output_info << "#grid_info: " << std::endl;                                 // grid information;
    output_info << "num_grains: \t" << grain.num_grain << std::endl;
    output_info << "Nx: \t\t" << Nx << std::endl;
    output_info << "Ny: \t\t" << Ny << std::endl;
    output_info << "delaX: \t" << para_pf.deltaX << std::endl;
    output_info << "delaY: \t" << para_pf.deltaX << "\n\n";

    output_info << "#parameters of this PF model" << std::endl;                 // PF parameters:
    output_info << "deltaT: \t\t" << para_pf.deltaT << std::endl;
    output_info << "garma(J/m2): \t" << para_pf.garma << std::endl;
    output_info << "deltaE: \t\t" << para_pf.deltaE << std::endl;
    output_info << "K: \t\t\t" << para_pf.K << std::endl;
    output_info << "Qb: \t\t\t" << para_pf.Qb << std::endl;
    output_info << "B: \t\t\t" << para_pf.B << std::endl;
    output_info << "R: \t\t\t" << para_pf.R << std::endl;
    output_info << "thigma: \t\t" << para_pf.thigma << std::endl;
    output_info << "W: \t\t\t" << para_pf.W << std::endl;
    output_info << "a: \t\t\t" << para_pf.a << std::endl;
    output_info << "Mij: \t\t\t" << para_pf.Mij << std::endl;
    output_info << "M: \t\t\t" << para_pf.M << "\n\n";


    output_info << "#initial grain and size" << std::endl;                      // initial grain radius
    output_info << "radius: " << grain.rad_ini << std::endl;
}

// initialize each grain and write the information of each grain core
void init_grain(NET grid, char globalName[], static std::vector<struct GRAIN> phi[][Ny],
                std::vector<struct GRAIN> phi_n[][Ny]) {
    
    std::cout << "********** grain initialization *********" << std::endl;
    std::fstream outfile;
    outfile.open(globalName, std::ios::out|std::ios::app);
    outfile << "position(x, y): " << std::endl;

    // initinalize the base grain (id: N-1)
    GRAIN tgrain = { grid.num_grain - 1,1.0 };
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            phi[i][j].push_back(tgrain);
            phi_n[i][j].push_back(tgrain);
        }
    }
    // initilize the other grain (id: 0-----N-2)
    struct POS {                                                                // is used for recording the core position for each grain
        int px;
        int py;
    };
    std::vector<struct POS> pos;
    POS tpos = { 0,0 };
    for (int i = 0; i < grid.num_grain; i++) {
        pos.push_back(tpos);
    }
    
    srand((int)time(NULL));
    int px, py;
    for (int i = 0; i < grid.num_grain - 1; i++) {
        px = rand() % Nx;
        py = rand() % Ny;
        bool flag = true;
        for (int j = 0; j < i; j++) {
            int dif = (int)sqrt(pow(pos[j].px - px, 2) + pow(pos[j].py - py, 2));
            if (dif <= (2*grid.rad_ini + grid.rad_dif)) {
                flag = false;
                i--;
                break;
            }
        }
        //if (i == 0) {
        //    px = 0;
        //    py = 0;
        //}

        int pd[9][2] = { 0 };                                                   // for saving the periodic coordinate
        if (flag == true) {                                                     // add the phi value for other grain 
            for (int j = 0; j < Nx; j++) {
                for (int k = 0; k < Ny; k++) {
                    pd[0][0] = j;
                    pd[0][1] = k;
                    int nloop = 9;
                    if (px < grid.rad_ini || px > Nx - grid.rad_ini || py < grid.rad_ini || py > Ny - grid.rad_ini)
                        periodic(j, k, pd);
                    else
                        nloop = 1;                                              // if the (px, py) is in the inside of box, then without judging the periodic position

                    bool flag1 = false;
                    for (int m = 0; m < nloop; m++) {                           // nine positions (self, periodic)
                        if (fabs(pd[m][0] - px) > grid.rad_ini || fabs(pd[m][1] - py) > grid.rad_ini)
                            continue;
                        int dif = sqrt(pow(pd[m][0] - px, 2) + pow(pd[m][1] - py, 2));
                        if (dif <= grid.rad_ini) {
                            flag1 = true;
                            break;
                        }
                    }
                    if (flag1 == true) {
                        tgrain.idg = i;
                        tgrain.phi = 1.0;
                        phi[j][k].clear();                                      // the base grain should be removed, then add the new grain into 
                        phi_n[j][k].clear();
                        phi[j][k].push_back(tgrain);
                        phi_n[j][k].push_back(tgrain);
                    }
                }
            }
            pos[i].px = px;
            pos[i].py = py;
            outfile << i << ": (" << px << ", " << py << ")" << std::endl;
        }
    }
}

// write the phi value and file for training
void output_grid(char vtkName[], char traName[], int num_grain, int step, 
                 bool flag_vtk, bool flag_tra, 
                 PPF para, std::vector<struct GRAIN> phi_n[][Ny], 
                 std::vector<struct GRAIN>::iterator it_pn) {

    // output the .vtk file for paraview
    char fileName[200];
    char cstep[50];
    if (flag_vtk == true) {
        double** phi_sum = new double* [Nx];                                    // for saving the sum value of phi
        for (int i = 0; i < Nx; i++) {
            phi_sum[i] = new double[Ny];
            memset(phi_sum[i], 0, sizeof(double) * Ny);
        }
        
        for (int i = 0; i < Nx; i++) {                                          // calculating the sum value of each grid
            for (int j = 0; j < Ny; j++) {
                double sum = 0.0;
                for (it_pn = phi_n[i][j].begin(); it_pn != phi_n[i][j].end(); it_pn++)
                    sum += pow(it_pn->phi, 2);
                if (sum > 1.0)
                    sum = 1.0;
                phi_sum[i][j] = sum;
            }
        }

        std::fstream outvtk;                                                    // creating the files for saving .vtk
        sprintf_s(cstep, "%d", step);
        strcpy_s(fileName, strlen(vtkName) + 1, vtkName);
        strcat_s(fileName, strlen(fileName) + strlen(cstep) + 1, cstep);
        strcat_s(fileName, strlen(fileName) + 5, ".vtk");
        outvtk.open(fileName, std::ios::out);
        // std::cout << "vtkName: " << vtkName << " " << fileName << std::endl;

        outvtk << "# vtk DataFile Version 2.0, time for the evolution, " << (double)step*para.deltaT << std::endl;
        outvtk << fileName << std::endl;
        outvtk << "ASCII " << std::endl;
        outvtk << "DATASET STRUCTURED_GRID" << std::endl;
        outvtk << "DIMENSIONS " << Ny << " " << Nx << " " << 1 << std::endl;
        outvtk << "POINTS " << Nx * Ny * 1 << " float" << std::endl;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                outvtk << i << " " << j << " " << 1 << std::endl;
            }
        }

        outvtk << "POINT_DATA " << Nx * Ny * 1 << std::endl;
        outvtk << "SCALARS CON float 1" << std::endl;
        outvtk << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                outvtk << phi_sum[i][j] << "\t";
            }
        }
        outvtk.close();
        for (int i = 0; i < Nx; i++)                                            
            delete[]phi_sum[i];
        delete[]phi_sum;
    }
    
    // output the file for training
    if (flag_tra == true) {
        std::fstream outtra;                                                    // create files for saving .phi
        strcpy_s(fileName, strlen(traName) + 1, traName);
        sprintf_s(cstep, "%d", step);
        strcat_s(fileName, strlen(fileName) + strlen(cstep) + 1, cstep);
        strcat_s(fileName, strlen(fileName) + 5, ".phi");
        outtra.open(fileName, std::ios::out);
        // std::cout << "traName: " << traName << " " << fileName << std::endl;

        outtra << "# phase value of grain growth for PINN traning file, created by Zhang: 2022/10/04" << std::endl;
        outtra << "# evolution time is: " << (double)step * para.deltaT << std::endl;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                outtra << "(" << i << " " << j << ") [";
                for (it_pn = phi_n[i][j].begin(); it_pn != phi_n[i][j].end(); it_pn++) {
                    if (phi_n[i][j].size() == 1)
                        outtra << it_pn->idg;
                    if (phi_n[i][j].size() > 1) {
                        outtra << it_pn->idg;
                        if (it_pn != phi_n[i][j].end() - 1)
                            outtra << " ";
                    }
                }
                outtra << "] ";
                for (it_pn = phi_n[i][j].begin(); it_pn != phi_n[i][j].end(); it_pn++)
                    outtra << it_pn->phi << " ";
                outtra << std::endl;
            }
        }
        outtra.close();
    }
}

// obtain periodic boundary for 2D 
void periodic (int x, int y, int pd[][2]) {

    int idex[8][2] = { 1,0, -1,0,  0,1,  0,-1, 1,1,  1,-1,  -1,1,  -1,-1 };     // it determins add or reduce the dimensions
    for (int i = 0; i < 8; i++) {
        pd[i+1][0] = x + idex[i][0] * Nx;
        pd[i+1][1] = y + idex[i][1] * Ny;
    }
}

void ar_points(int x, int y, int &lx, int &rx, int &dy, int &uy) {
    lx = x - 1;
    rx = x + 1;
    dy = y - 1;
    uy = y + 1;
    if (x == 0)
        lx = Nx - 1;
    if (x == Nx - 1)
        rx = 0;
    if (y == 0)
        dy = Ny - 1;
    if (y == Ny - 1)
        uy = 0;
}

double find_phi(int idg, int x, int y, std::vector<struct GRAIN> phi[][Ny]) {
    std::vector<struct GRAIN>::iterator itp;
    for (itp = phi[x][y].begin(); itp != phi[x][y].end(); itp++) {
        if (itp->idg == idg) {
            return itp->phi;
        }
    }
    return 0.0;
}
