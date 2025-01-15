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

# Power-not-served and Excess-power-served cost idea

```python formatter
# @formatter:off
                                                                                                                              # 0 or counter to handle degeneracy (to be discussed)
lego.model.pSlackPrice = pyo.Param(lego.model.i, initialize=pd.DataFrame([(i, max(lego.model.pProductionCost.values()) * 100 + (0 * max(lego.model.pProductionCost.values()) / 10)) for counter, i in enumerate(lego.model.i)], columns=["i", "values"]).set_index("i"), doc='Price of slack variable')
# @formatter:on
```

# Tight and compact Unit-Commitment constraints

Check, what SD/SU (Shutdown and Startup Capability) means in the context of Germáns paper "Tight and Compact MILP
Formulation for the Thermal Unit Commitment Problem" (doi: 10.1109/TPWRS.2013.2251373)

# Ramping constraints

Check eThRampUp and eThRampDown in the context of Germáns paper "Tight and Compact MILP Formulation for the Thermal Unit
Commitment Problem" (doi: 10.1109/TPWRS.2013.2251373)

He doesn't use vCommit in the formulation