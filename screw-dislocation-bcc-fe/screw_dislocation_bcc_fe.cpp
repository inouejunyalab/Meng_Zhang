//______________________________________________________
//  begin:  Thurs. Oct. 27, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//______________________________________________________
//------------------------------------------------------

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

const double pi = 3.14159265358979;
const double fe_lattice = 2.8553;

class Box {
public:
	static double orient[3][3];
	static void get_length_unitorient(double length_box[3], double unit_orient[3][3]);
};
double Box::orient[3][3] = { 1,1,-2, 1,-1,0, -1,-1,-1 };											// please specify the orientation (x,y,z)
void Box::get_length_unitorient(double length_box[3], double unit_orient[3][3]) {
	double dimension[3] = { 0 };
	double num_lattice[3] = { 22,38,0.5 };															// please specify the size of the box (x, y, z)
	for (int i = 0; i < 3; i++)	{
		dimension[i] = sqrt(pow(orient[i][0], 2) + pow(orient[i][1], 2) + pow(orient[i][2], 2));
		length_box[i] = num_lattice[i] * dimension[i] * fe_lattice;
	}
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			unit_orient[i][j] = orient[i][j] / dimension[i];
}

struct Coord {
	int id;
	int type;
	double x;
	double y;
	double z;
	Coord(int tid, int ttype, double tx, double ty, double tz) :id(tid), type(ttype), x(tx), y(ty), z(tz) 
	{};
};

// geting the Euler angle between original coordinate (100,010,001) with nuit_orient.
void get_euler_angle(double &alpha, double &beta, double &gramma, double unit_orient[3][3])	{		// transition from unit_orient to unit_matrix (100,010,001)
	std::cout << "finding the Euler angle: " << std::endl;

	double z1_proj_xoy = sqrt(pow(unit_orient[2][0], 2) + pow(unit_orient[2][1], 2));				// for testing the z{001} and z1{unit_orient[2][1-3]} collinear
	if (z1_proj_xoy > DBL_EPSILON) {
		double x1_1[3] = { unit_orient[2][1], -unit_orient[2][0], 0 };
		double x1_1_proj_x1 = x1_1[0] * unit_orient[0][0] + x1_1[1] * unit_orient[0][1] + x1_1[2] * unit_orient[0][2];
		double x1_1_proj_y1 = x1_1[0] * unit_orient[1][0] + x1_1[1] * unit_orient[1][1] + x1_1[2] * unit_orient[1][2];

		alpha = atan2(x1_1_proj_y1, x1_1_proj_x1);
		beta = atan2(z1_proj_xoy, unit_orient[2][2]);
		gramma = -atan2(x1_1[1], x1_1[0]);
	}
	else {
		alpha = 0.0;
		beta = (unit_orient[2][2] > 0.0) ? 0 : pi;
		gramma = -atan2(unit_orient[0][1], unit_orient[0][0]);
	}
	std::cout << alpha*180/pi << "(alpha) " << beta * 180 / pi << "(beta) " << gramma * 180 / pi << "(gramma) " << std::endl;
}
// rotation the coord vector
void rotation_euler(std::vector<struct Coord> &coord, std::vector<struct Coord>::iterator &itc, double &alpha, double &beta, double& gramma) {
	double middle = 0.0;																			// the matrix rotation from basic_orient(100,010,001) to unit_orient.
	middle = alpha;
	alpha = gramma;
	gramma = middle;

	double teuler[3] = { 0.0 };
	double euler_coord[3][3] = { 0.0 };
	euler_coord[0][0] = cos(gramma) * cos(alpha) - cos(beta) * sin(alpha) * sin(gramma);
	euler_coord[0][1] = cos(gramma) * sin(alpha) + cos(beta) * cos(alpha) * sin(gramma);
	euler_coord[0][2] = sin(gramma) * sin(beta);
	euler_coord[1][0] = -sin(gramma) * cos(alpha) - cos(beta) * sin(alpha) * cos(gramma);
	euler_coord[1][1] = -sin(gramma) * sin(alpha) + cos(beta) * cos(alpha) * cos(gramma);
	euler_coord[1][2] = cos(gramma) * sin(beta);
	euler_coord[2][0] = sin(beta) * sin(alpha);
	euler_coord[2][1] = -sin(beta) * cos(alpha);
	euler_coord[2][2] = cos(beta);

	for (itc = coord.begin(); itc != coord.end(); itc++) {
		for (int j = 0; j < 3; j++) {
			teuler[j] = euler_coord[j][0] * itc->x + euler_coord[j][1] * itc->y + euler_coord[j][2] * itc->z;
		}
		itc->x = teuler[0];
		itc->y = teuler[1];
		itc->z = teuler[2];
	}
}
// creating the simulation model for Iron
void building_matrix(std::vector<struct Coord> &coord, std::vector<struct Coord>::iterator &itc, double length_box[3], double unit_orient[3][3]) {
	std::vector<struct Coord> tcoord;
	std::vector<struct Coord>::iterator itt;
	double alpha = 0.0, beta = 0.0, gramma = 0.0;
	get_euler_angle(alpha, beta, gramma, unit_orient);

	Coord basic_atom1(1, 1, length_box[0]/2.0, length_box[1] / 2.0, length_box[2] / 2.0);
	Coord basic_atom2(2, 1, basic_atom1.x + fe_lattice / 2.0, basic_atom1.y + fe_lattice / 2.0, basic_atom1.z + fe_lattice / 2.0);		// old coordinate of atoms 2
	coord.push_back(basic_atom1);
	coord.push_back(basic_atom2);
	for (int i = 0; i < 3; i++) {	
		tcoord.clear();
		std::vector<struct Coord>().swap(tcoord);
		int count[3] = { 0.0, 0.0, 0.0 };
		while (true) {
			for (itc = coord.begin(); itc != coord.end(); itc++) {
				Coord temp_atom(1, 1, 0, 0, 0);
				temp_atom.x = itc->x + count[0] * fe_lattice;
				temp_atom.y = itc->y + count[1] * fe_lattice;
				temp_atom.z = itc->z + count[2] * fe_lattice;
				tcoord.push_back(temp_atom);

				if (count[i] != 0) {
					temp_atom.x = itc->x - count[0] * fe_lattice;
					temp_atom.y = itc->y - count[1] * fe_lattice;
					temp_atom.z = itc->z - count[2] * fe_lattice;
					tcoord.push_back(temp_atom);
				}
			}
			if (count[i] > 30)																// this value can be changed to make atoms be full of the box
				break;
			else
				count[i]++;
		}
		coord.clear();
		std::vector<struct Coord>().swap(coord);
		for (itt = tcoord.begin(); itt != tcoord.end(); itt++)
			coord.insert(coord.end(), itt, itt + 1);
	}
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		itc->x = itc->x - length_box[0] / 2.0;
		itc->y = itc->y - length_box[1] / 2.0;
		itc->z = itc->z - length_box[2] / 2.0;
	}
	rotation_euler(coord, itc, alpha, beta, gramma);
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		itc->x = itc->x + length_box[0] / 2.0;
		itc->y = itc->y + length_box[1] / 2.0;
		itc->z = itc->z + length_box[2] / 2.0;
	}

	tcoord.clear();
	std::vector<struct Coord>().swap(tcoord);
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		if (itc->x >= 0 && itc->x <= length_box[0] && itc->y >= 0 && itc->y <= length_box[1] && itc->z >= 0 && itc->z <= length_box[2])
			tcoord.insert(tcoord.end(), itc, itc + 1);
	}
	coord.clear();
	std::vector<struct Coord>().swap(coord);
	int count_id = 1;
	for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {								// set the type of atoms
		double dis = sqrt(pow(itt->x - length_box[0] / 2, 2) + pow(itt->y - length_box[1] / 2, 2));
		itt->id = count_id++;
		itt->type = 1;
		if(dis > length_box[0]/ 2.0 - 10)
			itt->type = 2;
		coord.insert(coord.end(), itt, itt + 1);
	}
}
void screw_dislocation(std::vector<struct Coord> &, std::vector<struct Coord>::iterator &);
void print_matrix(double length_box[3], double unit_orient[3][3]);

// --------------- main function
int main() {

	double length_box[3] = { 0 };
	double unit_orient[3][3] = { 0 };
	Box::get_length_unitorient(length_box, unit_orient);
	print_matrix(length_box, unit_orient);
	std::vector<struct Coord> coord;
	std::vector<struct Coord>::iterator itc;
	std::vector<struct Coord> tcoord;
	std::vector<struct Coord>::iterator itt;
	building_matrix(coord, itc, length_box, unit_orient);									// creating the model 
		tcoord.clear();
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		tcoord.insert(tcoord.end(), itc, itc + 1);
	}
	//screw_dislocation(tcoord, itt);															// creating the model with screw dislocation							

	// writing the data file
	std::string filename = "./fe_screw.dat";
	std::fstream outfile;
	outfile.open(filename, std::ios::out);
	outfile << "#BCC Fe model for MD buding by Meng" << "\n";
	outfile << tcoord.size() << " " << "atoms" << "\n";
	outfile << 2 << " " << "atom types" << "\n";
	outfile << 0.0 << " " << length_box[0] << " xlo xhi" << "\n";
	outfile << 0.0 << " " << length_box[1] << " ylo yhi" << "\n";
	outfile << 0.0 << " " << length_box[2] << " zlo zhi" << "\n";
	outfile << "\n";
	outfile << "Atoms # atomic" << "\n";
	outfile << "\n";

	for (itt = tcoord.begin(); itt != tcoord.end(); itt++) {
		double dis = sqrt(pow(itt->x - length_box[0] / 2.0, 2) + pow(itt->y - length_box[1] / 2.0, 2) + pow(itt->z - length_box[2] / 2.0, 2));
		if (dis < 60)
			outfile << itt->id << " " << 1 << " " << itt->x << " " << itt->y << " " << itt->z << "\n";
		else
			outfile << itt->id << " " << 2 << " " << itt->x << " " << itt->y << " " << itt->z << "\n";
	}
}

void screw_dislocation(std::vector<struct Coord>& coord, std::vector<struct Coord>::iterator& itc) {
	int cen_id[3] = { 0 };
	double coord_three[3][3] = { 0.0 };
	std::cout << "Input id of the three atoms in the perfect structure (first two paralle atoms): " << std::endl;
	std::cin >> cen_id[0] >> cen_id[1] >> cen_id[2];
	
	for (int i = 0; i < 3; i++)	{
		for (itc = coord.begin(); itc != coord.end(); itc++) {
			if (itc->id == cen_id[i]) {
				coord_three[i][0] = itc->x;
				coord_three[i][1] = itc->y;
				coord_three[i][2] = itc->z; 
			}				
		}
 	}

	double coord_core[3] = { 0.0 };
	coord_core[0] = (coord_three[0][0] + coord_three[1][0]) / 2.0;
	coord_core[1] = coord_three[0][1] + (coord_three[2][1]-coord_three[0][1])/3.0;

	double distance = sqrt((double)2.0 / 3.0) * fe_lattice;
	for (itc = coord.begin(); itc != coord.end(); itc++) {
		double rot_vect[2] = { -itc->x + coord_core[0], -itc->y + coord_core[1] };					// final coord (used for the NEB calculating)
		if (rot_vect[1] >= 0.0)
			itc->z = itc->z + sqrt(3) * fe_lattice / 2.0 / (2.0 * pi) * atan2(rot_vect[1], rot_vect[0]);
		else {
			itc->z = itc->z + sqrt(3) * fe_lattice / 2.0 / (2.0 * pi) * (2 * pi + atan2(rot_vect[1], rot_vect[0]));
		}
	}
}

void print_matrix(double length_box[3], double unit_orient[3][3]) {
	std::cout << "The information of matrix are: " << std::endl;
	std::cout << "vector of matrix: " << std::endl;
	for (int i = 0; i < 3; i++)
		std::cout << Box::orient[i][0] << " " << Box::orient[i][1] << " " << Box::orient[i][2] << std::endl;
	std::cout << std::endl;

	std::cout << "unit of vector: " << std::endl;
	for (int i = 0; i < 3; i++)
		std::cout << unit_orient[i][0] << " " << unit_orient[i][1] << " " << unit_orient[i][2] << std::endl;
	std::cout << std::endl;

	std::cout << "length of matrix: ";
	for (int i = 0; i < 3; i++)
		std::cout << length_box[i] << " ";
	std::cout << std::endl << std::endl;
}