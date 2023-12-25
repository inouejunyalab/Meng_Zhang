#include <fstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <io.h>
#include <direct.h>
#include "stgb_b.h"

CRY_BOX::CRY_BOX(double xyz[][3], double len_xyz[], double la_constant) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            xyz_vect[i][j] = xyz[i][j];
        }
        length_box[i] = len_xyz[i];
    }
    lattice = la_constant;
}

CRY_BOX::CRY_BOX(double xyz[][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            xyz_vect[i][j] = xyz[i][j];
        }
    }
}

// define the function in class
void CRY_BOX::get_euler_angle() {

	double z1_proj_xoy = sqrt(pow(xyz_vect[2][0], 2) + pow(xyz_vect[2][1], 2));			// for testing the z{001} and z1{matrix_orientation[2][1-3]} collinear
	if (z1_proj_xoy > DBL_EPSILON) {
		double x1_1[3] = { xyz_vect[2][1], -xyz_vect[2][0], 0 };
		double x1_1_proj_x1 = x1_1[0] * xyz_vect[0][0] + x1_1[1] * xyz_vect[0][1] + x1_1[2] * xyz_vect[0][2];
		double x1_1_proj_y1 = x1_1[0] * xyz_vect[1][0] + x1_1[1] * xyz_vect[1][1] + x1_1[2] * xyz_vect[1][2];

		euler_angle[0] = atan2(x1_1_proj_y1, x1_1_proj_x1);
		euler_angle[1] = atan2(z1_proj_xoy, xyz_vect[2][2]);
		euler_angle[2] = -atan2(x1_1[1], x1_1[0]);
	}
	else {
		euler_angle[0] = 0.0;
		euler_angle[1] = (xyz_vect[2][2] > 0.0) ? 0 : PI;
		euler_angle[2] = -atan2(xyz_vect[0][1], xyz_vect[0][0]);
	}

	double fai = euler_angle[2];
	double the = euler_angle[1];
	double psi = euler_angle[0];
	euler_matrix[0][0] = cos(psi) * cos(fai) - cos(the) * sin(fai) * sin(psi);
	euler_matrix[0][1] = cos(psi) * sin(fai) + cos(the) * cos(fai) * sin(psi);
	euler_matrix[0][2] = sin(psi) * sin(the);
	euler_matrix[1][0] = -sin(psi) * cos(fai) - cos(the) * sin(fai) * cos(psi);
	euler_matrix[1][1] = -sin(psi) * sin(fai) + cos(the) * cos(fai) * cos(psi);
	euler_matrix[1][2] = cos(psi) * sin(the);
	euler_matrix[2][0] = sin(the) * sin(fai);
	euler_matrix[2][1] = -sin(the) * cos(fai);
	euler_matrix[2][2] = cos(the);

	std::cout << "The euler angle are: " << fai * 180 / PI << " " << the * 180 / PI << " " << psi * 180 / PI << std::endl;
}

// euler_rotation
void euler_rotation(CRY_BOX matrix, std::vector<struct COORD>& coord,
					std::vector<struct COORD>::iterator& itc) {

	double teuler[3] = { 0.0 };
	double euler_coord[3][3] = { 0.0 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			euler_coord[i][j] = matrix.euler_matrix[i][j];
		}
	}
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		for (int i = 0; i < 3; i++) {
			teuler[i] = euler_coord[i][0] * itc->xyz[0] + euler_coord[i][1] * itc->xyz[1] + euler_coord[i][2] * itc->xyz[2];
		}
		for (int i = 0; i < 3; i++) {
			itc->xyz[i] = teuler[i];
		}
	}
}


// creating the model
void build_crystal(CRY_BOX matrix, std::vector<struct COORD>& coord,
				   std::vector<struct COORD>::iterator& itc, int atom_type) {

	std::vector<struct COORD> tcoord;
	std::vector<struct COORD>::iterator itt;
	double length_box[3];
	double la_const = matrix.lattice;
	double max_length = matrix.length_box[0];
	for (int i = 0; i < 3; i++) {
		length_box[i] = matrix.length_box[i];
		if (max_length < matrix.length_box[i])
			max_length = matrix.length_box[i];
	}
	std::cout << "the max length of the box is: " << max_length << std::endl;

	COORD atom1 = { 1, atom_type, 0.0, 0.0, 0.0 };
	COORD atom2 = { 2, atom_type, atom1.xyz[0] + la_const / 2.0, atom1.xyz[1] + la_const / 2.0, atom1.xyz[2] + la_const / 2.0 };		// old coordinate of atoms 2				// old coordinate of atoms 2

	tcoord.push_back(atom1);
	tcoord.push_back(atom2);
	int atom_id = 1;
	for (int bi = 0; bi < 3; bi++) {													// along x, y, z directions
		coord.clear();
		int count[3] = { 0,0,0 };
		while (true) {
			for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {
				COORD temp_atom = { atom_id, atom_type, 0, 0, 0 };
				for (int i = 0; i < 3; i++) {
					temp_atom.xyz[i] = itt->xyz[i] + count[i] * la_const;
				}
				coord.push_back(temp_atom);
				atom_id++;

				if (count[bi] != 0) {
					temp_atom.id = atom_id;
					for (int i = 0; i < 3; i++) {
						temp_atom.xyz[i] = itt->xyz[i] - count[i] * la_const;
					}
					coord.push_back(temp_atom);
					atom_id++;
				}
			}
			if (count[bi] > 30)															// this value can be changed to make the atoms be full of the box
				break;
			else
				count[bi]++;
		}
		tcoord.clear();
		for (itc = coord.begin(); itc != coord.end(); itc++)
			tcoord.insert(tcoord.end(), itc, itc + 1);
	}
	// rotating acoording to the Euler_matrix
	std::cout << "atoms... " << tcoord.size() << std::endl;
	for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {
		for (int i = 0; i < 3; i++) {
			itt->xyz[i] = itt->xyz[i] - length_box[i] / 2.0;
		}
	}
	euler_rotation(matrix, tcoord, itt);												// rotate the matrix 
	for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {
		for (int i = 0; i < 3; i++) {
			itt->xyz[i] = itt->xyz[i] + length_box[i] / 2.0;
		}
	}

	// cutting the model
	coord.clear();
	for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {
		if (itt->xyz[0] >= -1.0 && itt->xyz[0] <= matrix.length_box[0] + 1.0 &&
			itt->xyz[1] >= 0.0 && itt->xyz[1] <= matrix.length_box[1] &&
			itt->xyz[2] >= 0.0 && itt->xyz[2] <= matrix.length_box[2]) {
			coord.insert(coord.end(), itt, itt + 1);
		}
	}
}

void symm_crystal(CRY_BOX matrix, std::vector<struct COORD> &coord, std::vector<struct COORD>::iterator& itc) {
	std::vector<struct COORD> tcoord;
	std::vector<struct COORD>::iterator itt;

	double length_box_x = matrix.length_box[0];
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		COORD temp_atom = { 1, 2, 0, itc->xyz[1], itc->xyz[2] };
		temp_atom.xyz[0] = 2.0 * length_box_x - itc->xyz[0];
		tcoord.push_back(temp_atom);
	}
	for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {
		coord.insert(coord.end(), itt, itt + 1);
	}
}

// write the atoms into the files
void write_crystal(char file_name[], double length_box[], int types,
				   std::vector<struct COORD>& coord,
				   std::vector<struct COORD>::iterator& itc) {

	std::fstream outfile;
	outfile.open(file_name, std::ios::out);

	outfile << "#BCC Fe model for MD buding by Meng" << "\n";
	outfile << coord.size() << " " << "atoms" << "\n";
	outfile << types << " " << "atom types" << "\n";
	outfile << 0.0 << " " << length_box[0] << " xlo xhi" << "\n";
	outfile << 0.0 << " " << length_box[1] << " ylo yhi" << "\n";
	outfile << 0.0 << " " << length_box[2] << " zlo zhi" << "\n";
	outfile << "\n";
	outfile << "Atoms # atomic" << "\n";
	outfile << "\n";

	int count = 1;
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		outfile << count++ << " " << itc->type << " " << itc->xyz[0] << " " << itc->xyz[1] << " " << itc->xyz[2] << "\n";
	}
	outfile.close();
}
