//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Fri. Oct. 07, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//______________________________________________________
//------------------------------------------------------

#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <io.h>
#include <direct.h>
#include <stdlib.h>
#include "gg_dec.h"

// creating the necessary folder for saving the the all values
void build_folder(std::string &folderPath, char vtkName[], 
                  char traName[], char globalName[]) {

    // get the system time
    char buf_time[64];
    sys_time(buf_time);
    
    // the folderPath
    int i = 0, success = 0;
    while (buf_time[i] !=':') {
        folderPath += buf_time[i];
        i++;
    }
    if (0 != _access_s(folderPath.c_str(), 0))                                                      // c_str() return a pointer of the string
        success =_mkdir(folderPath.c_str());  
    std::cout << "Phase Field model for grain growth on:  " << buf_time << std::endl;

    // build folder and files
    char vtk_trafolder[2][20] = { "/output_vtk", "/output_tra" };
    char vtk_trafile[2][50] = { "/grain", "/train" };
    char global_file[50] = { "/grain_growth_steel.txt" };
    if (success != -1) {
        // vtk, train folder
        strcpy_s(vtkName, strlen(folderPath.c_str()) + 1, folderPath.c_str());
        strcpy_s(traName, strlen(folderPath.c_str()) + 1, folderPath.c_str());
        strcat_s(vtkName, strlen(folderPath.c_str()) + strlen(vtk_trafolder[0]) + 1, vtk_trafolder[0]);
        strcat_s(traName, strlen(folderPath.c_str()) + strlen(vtk_trafolder[1]) + 1, vtk_trafolder[1]);

        if (0 != _access_s(vtkName, 0))
            _mkdir(vtkName);
        if (0 != _access_s(traName, 0))
            _mkdir(traName);
        strcat_s(vtkName, strlen(vtkName) + strlen(vtk_trafile[0]) + 1, vtk_trafile[0]);            // should + 1 for the last charact '\0'
        strcat_s(traName, strlen(traName) + strlen(vtk_trafile[1]) + 1, vtk_trafile[1]);

        // global file name
        strcpy_s(globalName, strlen(folderPath.c_str()) + 1, folderPath.c_str());
        strcat_s(globalName, strlen(folderPath.c_str()) + strlen(global_file) + 1, global_file);
    }
}

// testing some operation of vector and the iterator
void test() {
    std::vector<int> test;
    std::vector<int> test2;
    std::vector<int>::iterator itt;
    std::vector<int>::iterator itt2;
    for (int i = 0; i < 5; i++) {
        test.push_back(i);
        test2.push_back(i);
    }

    for (itt = test.begin(), itt2 = test2.begin(); itt != test.end(); itt++) {
        std::cout << "before: " << &itt2 << " " << *itt2 << " " << *itt << std::endl;
        if (*itt2 == 4) {
            itt2 = test2.erase(itt2);
        }
        else {
            std::cout << "after: " << &itt2 << " " << *itt2 << std::endl;
            itt2++;
        }
    }
    for (int i = 0; i < test2.size(); i++)
        std::cout << "after erase2... " << test2.size() << " " << test2[i] << std::endl;
        test.erase(test.begin() + test.size(), test.end());
    for (itt = test.begin(); itt != test.end(); itt++)
        std::cout << "after erase... " << test.size() << " " << *itt << std::endl;

}

// main function for evolution of grain growth
int main() {
    //test();
    // define vector for saving values, thses vectors cannot be use the normal memory (stack, it is different from the heap memory),
    // it don't has enough memory size for saving these value, thsu, we define these into the static memory
    static std::vector<struct GRAIN> phi[Nx][Ny];
    static std::vector<struct GRAIN>::iterator itp;
    static std::vector<struct GRAIN> phi_n[Nx][Ny];
    static std::vector<struct GRAIN>::iterator it_pn;

    // creating folder accroding to the time (just: Y-M-D)
    std::string folderPath = { "./" };
    char vtkName[100], traName[100], globalName[100];
    build_folder(folderPath, vtkName, traName, globalName);

    // initial setting and wirte the information of grid, grain and, parameters
    NET grain(4, 10, 20);                                                                           // num_grain, rad_ini, rad_diff
    PPF para_pf;    
    OUTPUT out = {0, 50, 1};                                                                        // step counter, step_vtk, step_train
    output_info(grain, para_pf, out, globalName);
    init_grain(grain, globalName, phi, phi_n);                                                      // seting the initilization position

    // grain growht evolution
    while (true) {
        bool flag_vtk = false;
        bool flag_tra = false;
        if (out.step == 0) {
            std::cout << "\n" << "#starting evolution:" << std::endl;
            output_grid(vtkName, traName, grain.num_grain, 0, true, true, para_pf, phi_n, it_pn);   // output the initilizaiton configuration
        }
        else 
            std::cout << "steps: " << out.step << std::endl;
        
        int lx, rx, dy, uy;                                                                         // for saving the around points 
        std::vector<struct GRAIN>::iterator ittp;
        for (int ix = 0; ix < Nx; ix++) {
            for (int iy = 0; iy < Ny; iy++) {

                ar_points(ix, iy, lx, rx, dy, uy);
                for (itp = phi[ix][iy].begin(), it_pn = phi_n[ix][iy].begin(); itp != phi[ix][iy].end(); itp++) {

                    int idgi = itp->idg;
                    double dphi_dt = 0.0;
                    double lap = 0.0;
                    for (ittp = phi[ix][iy].begin(); ittp != phi[ix][iy].end(); ittp++) {
                        int idgk = ittp->idg;
                        double deltaE = 0.0;
                        if (idgi == grain.num_grain - 1 && idgk != grain.num_grain - 1)
                            deltaE = -0.24e6;
                        if (idgi != grain.num_grain - 1 && idgk == grain.num_grain - 1)
                            deltaE = +0.24e6;

                        lap = ((find_phi(idgk, lx, iy, phi) + find_phi(idgk, rx, iy, phi) +
                            find_phi(idgk, ix, dy, phi) + find_phi(idgk, ix, uy, phi) - 4 * find_phi(idgk, ix, iy, phi)) -
                            (find_phi(idgi, lx, iy, phi) + find_phi(idgi, rx, iy, phi) +
                                find_phi(idgi, ix, dy, phi) + find_phi(idgi, ix, uy, phi) - 4 * find_phi(idgi, ix, iy, phi))) /
                            pow(para_pf.deltaX, 2);
                        
                        dphi_dt += (2 * para_pf.M) * (para_pf.W * (find_phi(idgk, ix, iy, phi) - find_phi(idgi, ix, iy, phi)) +
                                    0.5 * pow(para_pf.a, 2) * lap -
                                    8.0 / PI * sqrt(find_phi(idgi, ix, iy, phi) * find_phi(idgk, ix, iy, phi)) * deltaE);
                    }
                    double delta_phi = -dphi_dt * para_pf.deltaT / phi[ix][iy].size();

                    if (itp->phi == 0.0 && delta_phi < 0.0) {                                       // delete the grain that has small dela_phi
                        it_pn = phi_n[ix][iy].erase(it_pn);                                         // if the delete value is the last one, then the iterator cannnot find
                    }
                    else {
                        it_pn->phi = delta_phi + itp->phi;
                        if (it_pn->phi < 10e-5)
                            it_pn->phi = 0.0;
                        if (it_pn->phi > 1.0)
                            it_pn->phi = 1.0;
                        it_pn++;
                    }
                }

                // sort the phi_n according to the phi value, and remove the grain with id larger than 3
                if (phi_n[ix][iy].size() > 3) {
                    sort(phi_n[ix][iy].begin(), phi_n[ix][iy].end(), LessSort);
                    phi_n[ix][iy].erase(phi_n[ix][iy].begin() + 3, phi_n[ix][iy].end());            
                }

                // normalization
                double phi_sum = 0.0;
                for (it_pn = phi_n[ix][iy].begin(); it_pn != phi_n[ix][iy].end(); it_pn++)
                    phi_sum += it_pn->phi;
                for (it_pn = phi_n[ix][iy].begin(); it_pn != phi_n[ix][iy].end(); it_pn++)
                    it_pn->phi = it_pn->phi / phi_sum;

                // find the the grains that are not in phi_n
                std::vector<int> dele_id;
                std::vector<int>::iterator itd;
                for (int i = 0; i < grain.num_grain - 1; i++) {
                    bool flag = true;
                    for (it_pn = phi_n[ix][iy].begin(); it_pn != phi_n[ix][iy].end(); it_pn++) {
                        if (i == it_pn->idg) {
                            flag = false;
                            break;
                        }
                    }
                    if (flag == true)
                        dele_id.push_back(i);
                }

                // add the grain(doesn't belong to phi_n) into phi_n, but the phi of the grain in the around point is not zero 
                int ar_point[4][2] = { lx,iy, rx,iy, ix,dy, ix,uy };                                // four around points of (ix, iy)
                GRAIN tgrain = { 0, 0.0 };
                for (itd = dele_id.begin(); itd != dele_id.end(); itd++) {
                    bool flag = false;
                    for (int i = 0; i < 4; i++) {
                        int tx = ar_point[i][0];
                        int ty = ar_point[i][1];

                        for (itp = phi[tx][ty].begin(); itp != phi[tx][ty].end(); itp++) {
                            if (itp->idg == *itd && itp->phi != 0) {
                                tgrain.idg = *itd;
                                phi_n[ix][iy].push_back(tgrain);
                                flag = true;
                                break;
                            }
                        }
                        if (flag == true)
                            break;
                    }
                }   // add the grain 
            }   // iy
        }   // ix

        // update the value in phi by phi_n
        int baseNum = 0;
        for (int ix = 0; ix < Nx; ix++) {
            for (int iy = 0; iy < Ny; iy++) {
                phi[ix][iy].clear();
                for (it_pn = phi_n[ix][iy].begin(); it_pn != phi_n[ix][iy].end(); it_pn++) {
                    phi[ix][iy].insert(phi[ix][iy].end(), it_pn, it_pn + 1);
                    if (it_pn->idg == grain.num_grain - 1 && it_pn->phi > 0.0)                      // checking the grid number for base grain (num_grain - 1)
                        baseNum++;
                }
            }
        }
        if ((double)baseNum / (double)(Nx * Ny) < 0.0001) {
            output_grid(vtkName, traName, grain.num_grain, out.step, true, true, para_pf, phi_n, it_pn);    // when this 
            break;
        }

        // output the vtk file and train files
        out.step++;
        if (out.step % out.stp_vtk == 0)
            flag_vtk = true;
        if (out.step % out.stp_tra == 0)
            flag_tra = true;
        output_grid(vtkName, traName, grain.num_grain, out.step, flag_vtk, flag_tra, para_pf, phi_n, it_pn);
    } // while()

    return 0;
}
