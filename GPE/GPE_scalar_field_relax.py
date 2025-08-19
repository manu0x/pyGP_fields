import numpy as np

class GPE_scalar_field_relax:
    '''This class stores the variables for a complex scalar field and necessary utilities for solving associated GPE-like PDE using ImEx RK methods '''
    def __init__(self,dim,N,im_rhs=None,ex_rhs=None,imx=None,ini_psi=None,relax=False):
        '''Initializer function, takes following arguments:
          1) dim->Dimension of space(1d, 2d or 3d)
          2) N-> no. of grid points along each direction
          3) im_rhs-> function that gives the linear rhs of PDE that is to be solved implicitly
          4) ex_rhs-> function that gives the remaining rhs terms, that to be solved explicilty
          5) imx-> is an object of class ImEx holding all coefficients of ImEx RK scheme to be used
          6) ini_psi-> Initial condition of psi, this is given as a complex np array of shape (N,)^dim i.e for 3d case it is of shape (N,N,N). The dtype should be complex even if the function is purely real.
          
          Please note that by design the 1st argument to im_rhs func is FT(psi) and for ex_rhs the 1st argument is psi itself. Rest of arguments are passed as a common argument-list
           which consist of all the arguments passed args in update_K function below. 
          '''
        #self.dx = dx

        self.my_shape=()
        self.s = imx.s
        '''s is no. of stages in ImEx RK method'''

        '''Relaxation related'''
        self.relax=relax
        self.rel_gamma = 1.0
        self.rel_num_sum = 0.0
        self.rel_den_sum = 0.0

        self.term1=0.0
        self.term2=0.0
        self.term3=0.0
        
        for i in range(dim):
            self.my_shape=self.my_shape+(N,)
        self.K_shape = (self.s,)+self.my_shape
        '''my_shape is the shape of psi e.g. in 2-d case with N grid points in each direction it will be (N,N)
            while K_shape is the shape of K arrays which store the contributions from individual stages hence their shape is like e.g. 2-d case with N gridpoints is [s,N,N]'''
        #print("class shapes",self.my_shape,self.K_shape)
        
        
        
        self.psi = np.zeros(self.my_shape,dtype=np.complex128)
        self.psi = 1.0*ini_psi

        self.mass_ini = np.sum(np.abs(ini_psi)**2) 
        '''In GP fields mass as defined as sum of |\psi|^2 over all the gridpoints is conserved '''
        
        #print(self.my_shape,self.psi.shape)
        '''f is the intermediate psi_k while f_t is its fourier transform'''
        self.f = np.zeros_like(self.psi)
        self.f_t = np.zeros_like(self.f)
        
        self.f = 1.0*self.psi
        
        self.im_K = np.zeros(self.K_shape,dtype=np.complex128)
        self.ex_K = np.zeros_like(self.im_K)
        
        
        self.ex_rhs = ex_rhs
        self.im_rhs = im_rhs
        
        self.im_A = imx.im_A
        self.ex_A = imx.ex_A
        
        self.im_B = imx.im_B
        self.ex_B = imx.ex_B
        
        self.im_C = imx.im_C
        self.ex_C = imx.ex_C
    
    def do_fft(self,s_cntr,lmda,dt):
        '''This function does the fft on summed up contributions of all previous stages and then multiplies the FT vector f_t
            by needed factors to do implicit calculations and then does inverse fft. Arguments: s_cntr-> no of stage working on,lmda->consists of necessary xi factors
            from the format (1+i*dt*im_A[s][s]*lmda)*f_t = ft(rhs)'''
        self.f_t = np.fft.fftn(self.f,self.my_shape)
    
        #(1+i*dt*a[s][s]*lmda)*f_t = ft(rhs)
        self.f_t = self.f_t/(1.0+1j*dt*lmda*self.im_A[s_cntr][s_cntr])
        self.f = np.fft.ifftn(self.f_t,self.my_shape)
        #print("f shape",self.f_t.shape,self.f.shape)
        
    def update_stage_sum(self,s_cntr,dt):
        '''This function sums up contriutions from all the previous substages to calc. rhs, on which we do fft in do_fft() function, for implicit calc.'''
        for i in range(s_cntr):
            if(i==0):
                self.f = self.psi + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            else:
                self.f = self.f + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            
    def update_K(self,s_cntr,dt,*args):
        '''This function stores the contribution from particular stage into K vectors'''
        #print("sncntr ",s_cntr)
       # print("psi shape",self.psi.shape,"f shape",self.f.shape)
        if (s_cntr==0)and(self.relax):
            self.rel_num_sum=0.0
            self.term1 = 0.0
        self.ex_K[s_cntr,:] = self.ex_rhs(self.f,self.f_t,*args)
        self.im_K[s_cntr,:] = self.im_rhs(self.f_t,self.f,*args)
        if(self.relax):
        #    #self.rel_num_sum+= np.sum(np.abs(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*(self.f-self.psi)))
            self.rel_num_sum+= np.sum(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*(self.f)) +\
                            np.sum((self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*np.conj(self.f))
            self.term1+=( np.sum(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*(self.f))  +\
                            np.sum((self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*np.conj(self.f)))
            #self.rel_num_sum+=np.sum((self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr]).real *(self.f-self.psi).real)
            #print(np.abs(np.sum(np.conj(self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr])*(self.f-self.psi))),np.sum((self.ex_B[s_cntr]*self.ex_K[s_cntr]+ self.im_B[s_cntr]*self.im_K[s_cntr]).real *(self.f-self.psi).real))

            
            
        
    def sum_contributions(self,dt):
        '''This function sums up the final contributions from all the stages weighted by respective coefficients(B(or b) from Butcher Tableau)'''
        term = np.zeros_like(self.psi)
        term_rel_den = 0.0
        term_rel_num = 0.0
        for i in range(self.s):
            term+=(dt*self.ex_B[i]*self.ex_K[i]+ dt*self.im_B[i]*self.im_K[i])
            term_rel_den+= (self.ex_B[i]*self.ex_K[i]+ self.im_B[i]*self.im_K[i])

            #for ii in range(self.s):
            #    term_rel_num+=(self.ex_A[i][ii]*self.ex_K[ii]+ self.im_A[i][ii]*self.im_K[ii])
            
            
        
        if (self.relax):
            self.rel_den_sum = (np.sum(np.conj(term_rel_den)*term_rel_den))
            #self.rel_num_sum = np.sum(np.conj(term_rel_den)*term_rel_num) + np.sum(term_rel_den*np.conj(term_rel_num))
            #self.term2 = np.sum(np.conj(term_rel_den)*term_rel_num) + np.sum(term_rel_den*np.conj(term_rel_num))
            self.term3 = (np.sum(np.conj(term_rel_den)*self.psi+np.conj(self.psi)*term_rel_den))
            #self.rel_num_sum = np.sum(np.conj(term_rel_den)*term_rel_num)
            #print("dndn",self.rel_den_sum,self.rel_num_sum)

            self.rel_gamma = 1.0*np.abs(-self.term3/(dt*self.rel_den_sum))#np.abs(2.0*self.rel_num_sum/self.rel_den_sum)

            #print("rrRel gamma is ",self.rel_gamma,self.rel_num_sum/self.rel_den_sum,self.rel_den_sum,self.rel_num_sum)
                
        mass_old = np.sum(np.conj(self.psi)*self.psi)

        self.psi = self.psi + self.rel_gamma*term
        mass_new = np.sum(np.conj(self.psi)*self.psi)
        #print("Mass change",np.abs(mass_old-mass_new), np.imag( self.rel_den_sum),self.rel_gamma)
    

            
        self.f = 0.0+self.psi
        
    def calc_mass(self):
        '''This calculates mass : sum |\psi|^2 over whole grid'''
        return(np.sum(np.abs(self.psi)**2).flatten())
    

        
       
        
       






