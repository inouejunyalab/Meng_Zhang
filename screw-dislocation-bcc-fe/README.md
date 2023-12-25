## Introduce screw dislocaiton into Fe:

This package is used to create the bcc fe mode with dislocation. 
The method used in the codes is the elastic deisplacement field (EDF): 𝑈_𝑧=𝑏/2𝜋 [𝑡𝑎𝑛]^(−1) (z/𝑥)

## How to use:
1) please specify three vectors of the orientation of the box and the number of duplication of unit cell on lines 28 and 31, respectively, acoordig to your needs.
2) creating the perfect matrix by annotating the function of "screw_dislocation(tcoord, itc)" on line 189.
3) chosing three atoms from the perfect matrix (the first are two parallel with the axis, and the third one is located on the vetex), which determin the dislocation position.
4) creating the model with screw dislcoation by removing the annotation of "screw_dislocation(tcoord, itc)" on line 189.

## License:
Copyright (C) 2022 Meng Zhang (meng_zhang@metall.t.u-tokyo.ac.jp) and Junya Inoue (junya_inoue@metall.t.u-tokyo.ac.jp). 
The __screw_dislocation_bcc_fe__ source code is distributed under a Mozilla Public License, V. 2.0.
