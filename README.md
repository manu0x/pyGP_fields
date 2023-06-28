# pyGP_fields
ImEx implementation for Gross-Pitavskii like fields.
Here we define few classes for implementation of ImEx methods for Gross–Pitaevskii like equations.
By Gross-Pitavskii like we mean partial differential equations of following type:
$$\frac{\partial \psi}{\partial t} = i \alpha(\nabla^n\psi) + f(\psi,t,\vec{x}) $$
where $\alpha$ is not dependent on space-variables($\vec{x}$), but it can depend on $t$.

We use (diagonally) Implicit-Explicit(ImEx) Runge-Kutta(RK) time stepping schemes with pseudospectral space discretization. For a quick introduction to both of these 
topics(and further resources), please see [David Ketcheson's short course](https://github.com/ketch/PseudoSpectralPython/). We borrow few examples 
from [David's repo](https://github.com/ketch/PseudoSpectralPython/) and the code here draws inspirations from his code.

Below we outline basic steps for implementation of an ImEx scheme(in context of our notation here):

ImEx notation: we denote the RK coefficients with letters $A$, $B$ and $C$ with following correspondence with Butcher Tableau:
```
  |
C |     A
  |
------------
  |    B
```
For ImEx, we use superscript to denote if this set of coefficients belongs to implicit or explicit table of ImEx method. For example $A_{ij}^{ex}$ is $(i,j)th$
component of $A$ of explicit tableau. A [ImEx](https://github.com/manu0x/pyGP_fields/blob/main/GPE/ImEx.py) class object holds these coefficients in the 
variables ```im_A,ex_A,im_B,ex_B,im_C,ex_C```. Please see [its read section](https://github.com/manu0x/pyGP_fields/blob/main/GPE/README.md) on how to
use that class.

If at current time($t$), solution is $\psi_t$ and we want to find solution $\psi_{t+dt}$ at time $t+dt$, we do: 
>For an ImEx scheme with $s$ stages:
>>At each stage i:
>>> $$\psi_k = \psi + (dt)\sum_{j=1}^{i-1}(A_{ij}^{im}K_j^{im} + A_{ij}^{ex}K_j^{ex})  + i(dt)A_{ii}^{im}\alpha\nabla^n\psi_k$$ 
        
>>>Defining, 
 >>>       $$f_r = \psi + (dt)\sum_{j=1}^{i-1}(A_{ij}^{im}K_j^{im} + A_{ij}^{ex}K_j^{ex})$$
>>>we get 
>>>        $$\psi_k = f_r  + i(dt)A_{ii}^{im}\alpha\nabla^n\psi_k$$
>>>Going to Fourier space, denoting corresponding quantities in Fourier space with hat($\hat{()}$):
>>>        $$\hat{\psi_k} = \frac{\hat{f_r}}{\left[1+i(dt)(A_{ii}^{im})\Lambda\right]}$$
>>>  where 
>>>        $$\Lambda\equiv -\alpha (ik)^n$$

>>>  We find $\psi_k$ by doing inverse FFT. Then we update $K_i^{im}$ and $K_i^{ex}$, by evaluating $i \alpha(\nabla^n\psi_k)$ and $f(\psi_k,t,\vec{x})$ i.e. 
>>>  given RHS of given PDE with $\psi_k$.

>>After $s$ stages, the contribution($K$) from all stages is summed over with respective weights($B$) of Butcher Tableau:
>>$$\psi_{t+dt} = \psi_{t} +(dt)\sum_{j=1}^{s}(B_{j}^{im}K_j^{im} + B_{j}^{ex}K_j^{ex})$$

The main class is [GPE_scalar_field](https://github.com/manu0x/pyGP_fields/blob/main/GPE/GPE_scalar_field.py) in [GPE](https://github.com/manu0x/pyGP_fields/tree/main/GPE).
It implements the above mentioned steps in a minimalistic way. Please see the associated information [there](https://github.com/manu0x/pyGP_fields/blob/main/GPE/) to see
how above steps are executed using 3 different functions in that class and how to use that class.

Here we provide examples from 1-d,2-d and 3-d cases on how to use these classes. All the example files in this repo have phrase "example" in their names along with dimensions.
For example [RSOC_2d_example](https://github.com/manu0x/pyGP_fields/blob/main/RSOC_2d_example.ipynb) is an example of 2-d Rasbha-Spin Orbit coupling case. 
        
