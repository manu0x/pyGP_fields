# GPE_scalar_field class


This class stores the arrays for the solution($\psi$) and intermediate solution ($\psi_k$).
Object for this class are created by doing a call like:

~~~
 psi = GPE_scalar_field(dim,N,im_rhs,ex_rhs,imx,psi_ini)
 ~~~

Here dim is dimensions of the space(e.g. 2 for 2d,3 for 3d etc), N is no. of gridpoints in each direction, im_rhs is a python function which gives linear rhs
of PDE(part that is to be handled implicitly) and ex_rhs for rhs to be dealt explicilty. psi_ini is the initial condition i.e. psi(t=t_ini).

Please see [this 1-d KdV example](https://github.com/manu0x/pyGP_fields/blob/main/KdV_Example.ipynb) for a demonstration on how to use it.




   
class **GPE\_scalar\_field**

[GPE\_scalar\_field](https://github.com/manu0x/pyGP_fields/blob/main/GPE/GPE_scalar_field.py)

  

[GPE\_scalar\_field](#GPE_scalar_field)(dim, N, im\_rhs=None, ex\_rhs=None, imx=None, ini\_psi=None)  
   
This class stores the variables for a complex scalar field and necessary utilities for solving associated GPE-like PDE using ImEx RK methods 

 

Methods defined here:  

**\_\_init\_\_**(self, dim, N, im\_rhs=None, ex\_rhs=None, imx=None, ini\_psi=None)

Initializer function, takes following arguments:  
1) dim->Dimension of space(1d, 2d or 3d)  
2) N-> no. of grid points along each direction  
3) im\_rhs-> function that gives the linear rhs of PDE that is to be solved implicitly  
4) ex\_rhs-> function that gives the remaining rhs terms, that to be solved explicilty  
5) imx-> is an [object](builtins.html#object) of class ImEx holding all coefficients of ImEx RK scheme to be used  
6) ini\_psi-> Initial condition of psi, this is given as a complex np array of shape (N,)^dim i.e for 3d case it is of shape (N,N,N). The dtype should be complex even if the function is purely real.  
   
Please note that by design the 1st argument to im\_rhs func is FT(psi) and for ex\_rhs the 1st argument is psi itself. Rest of arguments are passed as a common argument-list  
 which consist of all the arguments passed args in update\_K function below.

**calc\_mass**(self)

This calculates mass : sum |\\psi|^2 over whole grid

**do\_fft**(self, s\_cntr, lmda, dt)

This function does the fft on summed up contributions of all previous stages and then multiplies the FT vector f\_t  
by needed factors to do implicit calculations and then does inverse fft. Arguments: s\_cntr-> no of stage working on,lmda->consists of necessary xi factors  
from the format (1+i\*dt\*im\_A\[s\]\[s\]\*lmda)\*f\_t = ft(rhs)

**sum\_contributions**(self, dt)

This function sums up the final contributions from all the stages weighted by respective coefficients(B(or b) from Butcher Tableau)

**update\_K**(self, s\_cntr, \*args)

This function stores the contribution from particular stage into K vectors

**update\_stage\_sum**(self, s\_cntr, dt)

This function sums up contriutions from all the previous substages to calc. rhs, on which we do fft in [do\_fft](#GPE_scalar_field-do_fft)() function, for implicit calc.

* * *

Data descriptors defined here:  

**\_\_dict\_\_**

dictionary for instance variables (if defined)

**\_\_weakref\_\_**

list of weak references to the object (if defined)
