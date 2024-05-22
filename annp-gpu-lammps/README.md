## ANNP-GPU-Lammps (GPU-accelerated artificial neural network potential for molecular dynamics simulation)

## Description:
This package contains some GPU-implementations of artificial neural network potentials (ANNPs). The flexible computation approach (FCA) is used to increase performance (Reference 1). You can compile each potential into LAMMPS software (Reference 2). Please see the detailed procedures inside of each potential. 

Please contact us if you have any questions or suggestion for the implementation:
meng_zhang@metall.t.u-tokyo.ac.jp/mengzh90@gmail.com (M. Zhang), junya_inoue@metall.t.u-tokyo.ac.jp (J. Inoue) 

## Performance test:
The results of performance test are for "fe" and "ni" packages, which are carried out on Nvidia Quadro P5000 card.

## Reference:
1) M. Zhang, K, Hibi, J. Inoue, Comput. Phys. Commun. 108655, 285 (2023).
2) S. Plimpton, J. Comput. Phys. 117, 1 (1995).

## LAMMPS Version:
1) August 02 2023 stable

## License:
Copyright (C) 2022 Meng Zhang (meng_zhang@metall.t.u-tokyo.ac.jp), Koki Hibi (koki_hibi@metall.t.u-tokyo.ac.jp), Junya Inoue (junya_inoue@metall.t.u-tokyo.ac.jp).
The __annp-gpu-lammps__ source code is distributed under a Mozilla Public License, V. 2.0.
