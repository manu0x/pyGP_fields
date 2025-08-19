import numpy as np

class GPE_scalar_field_1d2c:
    def __init__(self,N,nc=2,im_rhs=None,ex_rhs=None,imx=None,ini_psi=None):
        self.my_shape=()
        self.s = imx.s
        self.dim = 1
        
        self.my_shape=(N,nc)
        self.K_shape = (self.s,)+self.my_shape

       # print("class shapes",self.my_shape,self.K_shape)
        
        
        
        self.psi = np.zeros(self.my_shape,dtype=np.complex128)
        self.psi = 1.0*ini_psi

        self.mass_ini = np.sum(np.abs(ini_psi)**2) 
        
        #print("my shape",self.my_shape,"psi shape",self.psi.shape)
        
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
        self.f_t[:,0] = np.fft.fft(self.f[:,0])
        self.f_t[:,1] = np.fft.fft(self.f[:,1])
    
        #(1+i*dt*a[s][s]*lmda)*f_t = ft(rhs)
        self.f_t = np.squeeze(np.matmul(lmda,np.expand_dims(self.f_t,axis=-1)))
        self.f[:,0] = np.fft.ifft(self.f_t[:,0])
        self.f[:,1] = np.fft.ifft(self.f_t[:,1])
        #print("f shape",self.f_t.shape,self.f.shape)
        
    def update_stage_sum(self,s_cntr,dt):
        for i in range(s_cntr):
            if(i==0):
                self.f = self.psi + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            else:
                self.f = self.f + dt*self.ex_A[s_cntr][i]*self.ex_K[i]+ dt*self.im_A[s_cntr][i]*self.im_K[i]
            
    def update_K(self,s_cntr,*args):
        #print("sncntr ",s_cntr)
       # print("psi shape",self.psi.shape,"f shape",self.f.shape)
        self.ex_K[s_cntr,:] = self.ex_rhs(self.f,self.f_t,*args)
        self.im_K[s_cntr,:] = self.im_rhs(self.f_t,self.f,*args)
        
    def sum_contributions(self,dt):
        for i in range(self.s):
            self.psi = self.psi + dt*self.ex_B[i]*self.ex_K[i]+ dt*self.im_B[i]*self.im_K[i]
            
        self.f = 0.0+self.psi
        
    def calc_mass(self):
        return(np.sum(np.abs(self.psi)**2).flatten())
    

        
        #im_rhs takes only fourier space vector 
        
       






