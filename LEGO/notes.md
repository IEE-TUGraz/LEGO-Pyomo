Checks and notes for future reference:

# vLineP-Limits

LEGO GAMS had the following limits for vLineP:

```gams
vLineP.up   (rpk(rp,k),i,j,c) $[le(i,j,c)]  =  pPmax (i,j,c) ;
vLineP.lo   (rpk(rp,k),i,j,c) $[le(i,j,c)]  = -pPmax (i,j,c) ;
vLineP.up   (rpk(rp,k),j,i,c) $[le(i,j,c)]  =  pPmax (i,j,c) ;
vLineP.lo   (rpk(rp,k),j,i,c) $[le(i,j,c)]  = -pPmax (i,j,c) ;
```

This created duplicate variables (since for each pair of nodes, there were two variables) and was not necessary. This
might however be necessary for SOCP, so please check again when implementing it.

# Power-not-served and Excess-power-served cost idea

```python formatter
# @formatter:off
                                                                                                                              # 0 or counter to handle degeneracy (to be discussed)
lego.model.pSlackPrice = pyo.Param(lego.model.i, initialize=pd.DataFrame([(i, max(lego.model.pProductionCost.values()) * 100 + (0 * max(lego.model.pProductionCost.values()) / 10)) for counter, i in enumerate(lego.model.i)], columns=["i", "values"]).set_index("i"), doc='Price of slack variable')
# @formatter:on
```

# Tight and compact Unit-Commitment constraints

The implementation of Unit-Commitment is based on Germáns paper "Tight and Compact MILP Formulation for the Thermal Unit
Commitment Problem" (doi: 10.1109/TPWRS.2013.2251373). It is however adjusted by assuming SD & SU (Shutdown and Startup
Capability) to be equal to the minimum of production as a simplification, which is the most conservative approach. This
was done also since the data is most probably not available for those detailed technical specifications.

# Ramping constraints

Check eThRampUp and eThRampDown in the context of Germáns paper "Tight and Compact MILP Formulation for the Thermal Unit
Commitment Problem" (doi: 10.1109/TPWRS.2013.2251373)

He doesn't use vCommit in the formulation

# Forcing investment in lines in specific order

LEGO-GAMS had the following constraint active to force investment in a specific order (first c, then c+1, and so on):

```gams
eTranInves (i,j,c) $[lc(i,j,c) and pEnableTransNet and ord(c)>1]..
   vLineInvest(i,j,c) =l= vLineInvest(i,j,c-1) + sum[le(i,j,c-1),1];
```

This is not implemented in LEGO-Pyomo because there might be cases where two options are available (e.g., restringing a
line vs. building a new one). Forcing an order would lead to wrong answers here. If the model returns degenerate
solutions, the issues can be overcome by slightly changing investment cost of otherwise identical options. 