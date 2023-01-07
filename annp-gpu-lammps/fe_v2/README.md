## This is new GPU implementation for ANNP of Fe (references 1 and 2), which has a 12-14% speedup increasing than the previous one, as can be seen from performance test. You can compile this into LAMMPS (reference 3) according to the procedures in "fe" file.

## Update:
1) removed redundant computation of symmetry functions 
2) modified the force updating for atom i

## Tested systems, GPU cards, Lammps version:
1) Ubuntu 16.04 (System)
2) Nvidia RTX A5000 (GPU card)
3) September 2021 stable version (Lammps)

## References:
1) H. Mori, T. Ozaki, Phys Rev Mater. 4, 040601(R) (2020).
2) N. Artrith and A. Urban, Comput. Mater. Sci. 114, 135 (2016).
3) S. Plimpton, J. Comput. Phys. 117, 1 (1995).

## Update and release:
[GitHub] (https://github.com/inouejunyalab/Meng_Zhang/tree/main/annp-gpu-lammps/fe_v2)

## Contact Information:
Email: meng_zhang@metall.t.u-tokyo.ac.jp (M. Zhang), junya_inoue@metall.t.u-tokyo.ac.jp (J. Inoue)
Please contact us if you have any questions or suggestion for the implementation

## License:
Copyright (C) 2022 Meng Zhang (meng_zhang@metall.t.u-tokyo.ac.jp), Koki Hibi (koki_hibi@metall.t.u-tokyo.ac.jp), Junya Inoue (junya_inoue@metall.t.u-tokyo.ac.jp).
The source code is distributed under a Mozilla Public License, V. 2.0.
