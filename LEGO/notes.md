Checks and notes for future reference:

# vLineP-Limits
LEGO GAMS had the following limits for vLineP:
```gams
vLineP.up   (rpk(rp,k),i,j,c) $[le(i,j,c)]  =  pPmax (i,j,c) ;
vLineP.lo   (rpk(rp,k),i,j,c) $[le(i,j,c)]  = -pPmax (i,j,c) ;
vLineP.up   (rpk(rp,k),j,i,c) $[le(i,j,c)]  =  pPmax (i,j,c) ;
vLineP.lo   (rpk(rp,k),j,i,c) $[le(i,j,c)]  = -pPmax (i,j,c) ;
```
This created duplicate variables (since for each pair of nodes, there were two variables) and was not necessary. 
Please check, if this might however be necessary for SOCP (or similar). 

Note: Additionally, bounds for candidate lines were added to make the model more tight. 