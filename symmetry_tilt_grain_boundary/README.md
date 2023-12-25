## This code is used to create the model with symmetry tilt grain boundary (STGB).

## How to using:
1) specify the vector along z-axis in line 21 of "stgb.cpp" file.
2) changing the length of box for one grain in line 22 of "stgb.cpp" file according to you requirement. If you want to create a large model, the number in line 133 of "stgb_b.cpp" need to be changed to make the atoms be full of the box.

## Note:
After creating the model, you need to delete the overlap atoms on two GBs.
