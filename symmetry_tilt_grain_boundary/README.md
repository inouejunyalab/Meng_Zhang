## This code is used to create the model with symmetry tilt grain boundary (STGB).

## How to using:
1) specify the vector along z-axis on line 21 in "stgb.cpp" file.
2) changing the length of box for one grain on line 22 in "stgb.cpp" file according to you needs. If you want to create a large model, the number on line 133 in "stgb_b.cpp" needs to be changed to make the atoms be full of the box.

## Note:
After creating the model, you need to delete the overlap atoms on two GBs.

## License:
Copyright (C) 2022 Meng Zhang (meng_zhang@metall.t.u-tokyo.ac.jp), Koki Hibi (koki_hibi@metall.t.u-tokyo.ac.jp), Junya Inoue (junya_inoue@metall.t.u-tokyo.ac.jp). The __symmetry_tilt_grain_boundary__ source code is distributed under a Mozilla Public License, V. 2.0.
