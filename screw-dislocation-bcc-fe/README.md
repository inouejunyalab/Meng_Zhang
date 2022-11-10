## Introduce screw dislocaiton into Fe

This package is used to create the bcc fe mode with dislocation. 
The method used in the codes is the elastic deisplacement field (EDF): 𝑈_𝑧=𝑏/2𝜋 [𝑡𝑎𝑛]^(−1) (z/𝑥)

If you want to create the dislocation accrding to your requirement, please note these:
1) specify the orientation (x,y,z) on line 32
2) specify size of the box along (x,y,z) on line 35
3) specify the position the disloction core on line 259, but please be careful about the EDF decomposition on line 262.
The code give us the structure dislocation line parallel with the y axis <1,-1,1> direciton
