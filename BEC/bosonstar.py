from .relaxation_method import *
from .parallel import parallel
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np

"""
    The primary task of this module is to extend the relaxation method in relaxation_method.py to solve the 
    equations of motion of the relativistic Einstein-Klein-Gordon equations for Bose-Einstein condensates (BECs).
    This :class: 'bosonstar' can be called directly if one wishes to modify the stucture of this module. 
    However, we provide several helper functions which may be called directly without modifying the class 
    and should return the solutions.

    :func: 'get_profiles' which returns the solutions of the equations of motion in a matrix.
    this function is necessary to iteravely solve the equations of motion until the ground state is found.

    :func: '_get_profiles' which is an experimental alternative to 'get_profiles' which performs better for large 
    mass ratios.

    :func: 'get_compactness' returns the mass and compactness of the Bose-Einstein condensate

    :func: 'get_density' returns the density of the Bose-Einstein condensate

    :func: 'scan' calls the helper functions to perfom the calculations for an array of central densities for the 
    scalars that make up the Bose-Einstein condensate using the multiprocessing module for faster calcuations 
"""

class bosonstar(relaxation_method):
    """
        The purpose of of this module is to extend the relaxation_method to solve the equations of motion 
        of the BEC system.  
       
        :func: init overides :func: __init__ in :class: relaxation_method.  

        :func: f overides :func: f in :class: relaxation_method. defines the differential equations of the 
        equation of motion

        :func: jac overides :func: jac in :class: relaxation_method. defines the jacobian of the 
        differential equations. 

        :func: findMBS find the total mass in solar units and the compactness of the BEC in SI units. also 
        returns the core radius and density of the BEC 

        The other functions in this class are necessary for the relaxation algorithm

        :func: dB1dy and dB2dy are the boundary conditions of the differential equations 
        :func: scale sets the scale of the numerical values for the relaxation algorithm

        Paramters:
        -----------
            lam1 : float
                Self-coupling of the the first scalar in the equations of motion
            lam2 : float
                Self-coupling of the the second scalar in the equations of motion
            lam12 : float
                Interaction-coupling between the scalars in the equations of motion
            m1 : float
                Mass of the first scalar in units of eV
            m2 : float
                Mass of the second scalar in units of eV
            fr : float
                Decay constant of the scalars in units of GeV. If set to a value, it normalizes the 
                couplings else it can be set to none and not impact the equations of motion.
                Defined to be uniform for all scalars

        Attributes:
        -----------
            mphi : float
                Mass of the first scalar converted to units of GeV
            mr : float
                Mass ratio of the scalars
            G_n : float
                Gravitational constant 
            c : float
                Sign of the mass ratio, + or -
            tol : float
                Tolerance of the relaxation method to determine the stoping criteria of the relaxation.
                Overrides parameter in relaxation_method which is set 1e-15
            slowc : float
                Parameter of the relaxation method. Probably does not need to be changed. Controls how 
                fast the solution converges.
    """
    def init(
        self,
        lam1=1,
        lam2=1,
        lam12=1,
        m1=1e-10,
        m2=1e-10,
        fr=1e17
    ):
        self.slowc = 1.0
        self.mphi = m1*1.e-9
        self.mr = m2/m1
        self.G_N = 6.70883e-39
        self.fr = fr 
        if fr is None:
            self.lam1 = lam1
            self.lam2 = lam2
            self.lam12 = lam12
        else:
            self.lam1 = lam1/(4*np.pi*self.G_N*self.fr**2) 
            self.lam2 = lam2/(4*np.pi*self.G_N*self.fr**2) 
            self.lam12 = lam12/(4*np.pi*self.G_N*self.fr**2) 
        self.c = np.sign(self.mr)
        self.tol = 1e-8

    def f(self,r,y):
        """
        Differential equations of the EKG system.
        Used in the relaxation algorithm.

        Parameters:
        -----------
            r : float 
                radial coordinate of the profiles
            y : numpy array 
                derivatives of the profiles (see below for the positions of each profile)
                y[0] = A
                y[1] = phi1
                y[2] = phi2
                y[3] = dphi1
                y[4] = dphi2
                y[5] = B 
                y[6] = mu1 
                y[7] = mu1 
        """
        A = y.item(0)
        phi1 = y.item(1)
        phi2 = y.item(2)
        dphi1 = y.item(3)
        dphi2 = y.item(4)
        B = y.item(5)
        mu1 = y.item(6)
        mu2 = y.item(7)
        lam1 = self.lam1 
        lam2 = self.lam2 
        lam12 = self.lam12
        c = self.c
        mr = self.mr

        f0 = A**2*(dphi1**2/A + dphi2**2/A + (c + mu1**2/B)*phi1**2 + (lam1*phi1**4)/2 + \
             (c*mr**2 + mu2**2/B)*phi2**2 + (1/2)*lam12*phi1**2*phi2**2 + (lam2*phi2**4)/2 - 1/r**2 + \
             1/(A*r**2))*r
        f1 = dphi1
        f2 = dphi2
        f3 = -((1/(2*B*r))*(2*B*dphi1 + 2*A*B*dphi1 - 2*A*B*c*phi1*r + 2*A*mu1**2*phi1*r - \
             2*A*B*lam1*phi1**3*r - A*B*lam12*phi1*phi2**2*r - 2*A*B*c*dphi1*phi1**2*r**2 - \
             A*B*dphi1*lam1*phi1**4*r**2 - 2*A*B*c*dphi1*mr**2*phi2**2*r**2 - \
             A*B*dphi1*lam12*phi1**2*phi2**2*r**2 - A*B*dphi1*lam2*phi2**4*r**2))
        f4 = -((1/(2*B*r))*(2*B*dphi2 + 2*A*B*dphi2 - 2*A*B*c*mr**2*phi2*r + 2*A*mu2**2*phi2*r - \
             A*B*lam12*phi1**2*phi2*r - 2*A*B*lam2*phi2**3*r - 2*A*B*c*dphi2*phi1**2*r**2 - \
             A*B*dphi2*lam1*phi1**4*r**2 - 2*A*B*c*dphi2*mr**2*phi2**2*r**2 - \
             A*B*dphi2*lam12*phi1**2*phi2**2*r**2 - A*B*dphi2*lam2*phi2**4*r**2))
        f5 = A*B*(dphi1**2/A + dphi2**2/A + (-c + mu1**2/B)*phi1**2 - (lam1*phi1**4)/2 + \
             ((-c)*mr**2 + mu2**2/B)*phi2**2 - (1/2)*lam12*phi1**2*phi2**2 - (lam2*phi2**4)/2 + \
             1/r**2 - 1/(A*r**2))*r
        f6 = 0
        f7 = 0

        f = [f0,f1,f2,f3,f4,f5,f6,f7]

        return np.array(f)

    def jac(self,r,y):
        """
        Jacobians of the Differential Equations of the EKG system.
        Used in the relaxation algorithm.

        Parameters:
        -----------
            r : float 
                radial coordinate of the profiles
            y : numpy array 
                derivatives of the profiles (see below for the positions of each profile)
                y[0] = A
                y[1] = phi1
                y[2] = phi2
                y[3] = dphi1
                y[4] = dphi2
                y[5] = B 
                y[6] = mu1 
                y[7] = mu1 
        """
        A = y.item(0)
        phi1 = y.item(1)
        phi2 = y.item(2)
        dphi1 = y.item(3)
        dphi2 = y.item(4)
        B = y.item(5)
        mu1 = y.item(6)
        mu2 = y.item(7)
        lam1 = self.lam1
        lam2 = self.lam2
        lam12 = self.lam12
        c = self.c
        mr = self.mr
        N = self.N
        dgdy = np.zeros((N,N))

        #A
        dgdy[0][0] = A**2*(-(dphi1**2/A**2) - dphi2**2/A**2 - 1/(A**2*r**2))*r + \
           2*A*(dphi1**2/A + dphi2**2/A + (c + mu1**2/B)*phi1**2 + (lam1*phi1**4)/2 + \
           (c*mr**2 + mu2**2/B)*phi2**2 + (1/2)*lam12*phi1**2*phi2**2 + (lam2*phi2**4)/2 - \
           1/r**2 + 1/(A*r**2))*r
        dgdy[0][1] = A**2*(2*(c + mu1**2/B)*phi1 + 2*lam1*phi1**3 + lam12*phi1*phi2**2)*r
        dgdy[0][2] = A**2*(2*(c*mr**2 + mu2**2/B)*phi2 + lam12*phi1**2*phi2 + \
           2*lam2*phi2**3)*r
        dgdy[0][3] = 2*A*dphi1*r
        dgdy[0][4] = 2*A*dphi2*r
        dgdy[0][5] = A**2*(-((mu1**2*phi1**2)/B**2) - (mu2**2*phi2**2)/B**2)*r
        dgdy[0][6] = (2*A**2*mu1*phi1**2*r)/B
        dgdy[0][7] = (2*A**2*mu2*phi2**2*r)/B
        #Phi1
        dgdy[1][0] = 0 
        dgdy[1][1] = 0
        dgdy[1][2] = 0
        dgdy[1][3] = 1
        dgdy[1][4] = 0 
        dgdy[1][5] = 0
        dgdy[1][6] = 0
        dgdy[1][7] = 0
        #Phi2
        dgdy[2][0] = 0
        dgdy[2][1] = 0
        dgdy[2][2] = 0
        dgdy[2][3] = 0
        dgdy[2][4] = 1
        dgdy[2][5] = 0
        dgdy[2][6] = 0
        dgdy[2][7] = 0
        #dPhi1
        dgdy[3][0] = -((1/(2*B*r))*(2*B*dphi1 - 2*B*c*phi1*r + 2*mu1**2*phi1*r - \
             2*B*lam1*phi1**3*r - B*lam12*phi1*phi2**2*r - 2*B*c*dphi1*phi1**2*r**2 - \
             B*dphi1*lam1*phi1**4*r**2 - 2*B*c*dphi1*mr**2*phi2**2*r**2 - \
             B*dphi1*lam12*phi1**2*phi2**2*r**2 - B*dphi1*lam2*phi2**4*r**2))
        dgdy[3][1] = -((1/(2*B*r))*(-2*A*B*c*r + 2*A*mu1**2*r - 6*A*B*lam1*phi1**2*r - \
             A*B*lam12*phi2**2*r - 4*A*B*c*dphi1*phi1*r**2 - 4*A*B*dphi1*lam1*phi1**3*r**2 - \
             2*A*B*dphi1*lam12*phi1*phi2**2*r**2))
        dgdy[3][2] = -((1/(2*B*r))*(-2*A*B*lam12*phi1*phi2*r - \
             4*A*B*c*dphi1*mr**2*phi2*r**2 -  2*A*B*dphi1*lam12*phi1**2*phi2*r**2 - \
             4*A*B*dphi1*lam2*phi2**3*r**2))
        dgdy[3][3] = -((1/(2*B*r))*(2*B + 2*A*B - 2*A*B*c*phi1**2*r**2 - \
             A*B*lam1*phi1**4*r**2 - 2*A*B*c*mr**2*phi2**2*r**2 - A*B*lam12*phi1**2*phi2**2*r**2 - \
             A*B*lam2*phi2**4*r**2))
        dgdy[3][4] = 0
        dgdy[3][5] = -((1/(2*B*r))*(2*dphi1 + 2*A*dphi1 - 2*A*c*phi1*r - \
              2*A*lam1*phi1**3*r - A*lam12*phi1*phi2**2*r - 2*A*c*dphi1*phi1**2*r**2 - \
              A*dphi1*lam1*phi1**4*r**2 - 2*A*c*dphi1*mr**2*phi2**2*r**2 - \
              A*dphi1*lam12*phi1**2*phi2**2*r**2 - A*dphi1*lam2*phi2**4*r**2)) + (1/(2*B**2*r))*(2*B*dphi1 + \
            2*A*B*dphi1 - 2*A*B*c*phi1*r + 2*A*mu1**2*phi1*r - 2*A*B*lam1*phi1**3*r - \
            A*B*lam12*phi1*phi2**2*r - 2*A*B*c*dphi1*phi1**2*r**2 - A*B*dphi1*lam1*phi1**4*r**2 - \
            2*A*B*c*dphi1*mr**2*phi2**2*r**2 - A*B*dphi1*lam12*phi1**2*phi2**2*r**2 - \
            A*B*dphi1*lam2*phi2**4*r**2)
        dgdy[3][6] = -((2*A*mu1*phi1)/B)
        dgdy[3][7] = 0 
        #dPhi2
        dgdy[4][0] = -((1/(2*B*r))*(2*B*dphi2 - 2*B*c*mr**2*phi2*r + 2*mu2**2*phi2*r - \
             B*lam12*phi1**2*phi2*r - 2*B*lam2*phi2**3*r - 2*B*c*dphi2*phi1**2*r**2 - \
             B*dphi2*lam1*phi1**4*r**2 - 2*B*c*dphi2*mr**2*phi2**2*r**2 - \
             B*dphi2*lam12*phi1**2*phi2**2*r**2 - B*dphi2*lam2*phi2**4*r**2))
        dgdy[4][1] = -((1/(2*B*r))*(-2*A*B*lam12*phi1*phi2*r - \
             4*A*B*c*dphi2*phi1*r**2 - 4*A*B*dphi2*lam1*phi1**3*r**2 - \
             2*A*B*dphi2*lam12*phi1*phi2**2*r**2))
        dgdy[4][2] = -((1/(2*B*r))*(-2*A*B*c*mr**2*r + 2*A*mu2**2*r - \
             A*B*lam12*phi1**2*r - 6*A*B*lam2*phi2**2*r - 4*A*B*c*dphi2*mr**2*phi2*r**2 - \
             2*A*B*dphi2*lam12*phi1**2*phi2*r**2 - 4*A*B*dphi2*lam2*phi2**3*r**2))
        dgdy[4][3] = 0
        dgdy[4][4] = -((1/(2*B*r))*(2*B + 2*A*B - 2*A*B*c*phi1**2*r**2 - \
             A*B*lam1*phi1**4*r**2 - 2*A*B*c*mr**2*phi2**2*r**2 - A*B*lam12*phi1**2*phi2**2*r**2 - \
             A*B*lam2*phi2**4*r**2))
        dgdy[4][5] = -((1/(2*B*r))*(2*dphi2 + 2*A*dphi2 - 2*A*c*mr**2*phi2*r - \
              A*lam12*phi1**2*phi2*r - 2*A*lam2*phi2**3*r - 2*A*c*dphi2*phi1**2*r**2 - \
              A*dphi2*lam1*phi1**4*r**2 - 2*A*c*dphi2*mr**2*phi2**2*r**2 - \
              A*dphi2*lam12*phi1**2*phi2**2*r**2 - A*dphi2*lam2*phi2**4*r**2)) + (1/(2*B**2*r))*(2*B*dphi2 + \
            2*A*B*dphi2 - 2*A*B*c*mr**2*phi2*r + 2*A*mu2**2*phi2*r - \
            A*B*lam12*phi1**2*phi2*r - 2*A*B*lam2*phi2**3*r - 2*A*B*c*dphi2*phi1**2*r**2 - \
            A*B*dphi2*lam1*phi1**4*r**2 - 2*A*B*c*dphi2*mr**2*phi2**2*r**2 - \
            A*B*dphi2*lam12*phi1**2*phi2**2*r**2 - A*B*dphi2*lam2*phi2**4*r**2)
        dgdy[4][6] = 0
        dgdy[4][7] = -((2*A*mu2*phi2)/B)
        #B
        dgdy[5][0] = A*B*(-(dphi1**2/A**2) - dphi2**2/A**2 + 1/(A**2*r**2))*r + \
           B*(dphi1**2/A + dphi2**2/A + (-c + mu1**2/B)*phi1**2 - (lam1*phi1**4)/2 + \
           ((-c)*mr**2 + mu2**2/B)*phi2**2 - (1/2)*lam12*phi1**2*phi2**2 - \
           (lam2*phi2**4)/2 + 1/r**2 - 1/(A*r**2))*r
        dgdy[5][1] = A*B*(2*(-c + mu1**2/B)*phi1 - 2*lam1*phi1**3 - lam12*phi1*phi2**2)*r
        dgdy[5][2] = A*B*(2*((-c)*mr**2 + mu2**2/B)*phi2 - lam12*phi1**2*phi2 - \
           2*lam2*phi2**3)*r
        dgdy[5][3] = 2*B*dphi1*r
        dgdy[5][4] = 2*B*dphi2*r
        dgdy[5][5] = A*B*(-((mu1**2*phi1**2)/B**2) - (mu2**2*phi2**2)/B**2)*r + \
          A*(dphi1**2/A + dphi2**2/A + (-c + mu1**2/B)*phi1**2 - (lam1*phi1**4)/2 + \
            ((-c)*mr**2 + mu2**2/B)*phi2**2 - (1/2)*lam12*phi1**2*phi2**2 - (lam2*phi2**4)/2 + 1/r**2 - \
            1/(A*r**2))*r
        dgdy[5][6] = 2*A*mu1*phi1**2*r
        dgdy[5][7] = 2*A*mu2*phi2**2*r
        #mu1
        dgdy[6][0] = 0
        dgdy[6][1] = 0
        dgdy[6][2] = 0
        dgdy[6][3] = 0
        dgdy[6][4] = 0
        dgdy[6][5] = 0
        dgdy[6][6] = 0
        dgdy[6][7] = 0
        #mu2
        dgdy[7][0] = 0
        dgdy[7][1] = 0
        dgdy[7][2] = 0
        dgdy[7][3] = 0
        dgdy[7][4] = 0
        dgdy[7][5] = 0
        dgdy[7][6] = 0
        dgdy[7][7] = 0
        return dgdy

    def dB1dy(self):
        """
        Boundary Conditions at r = 0.
        Used in relaxation algorithm.
        Shape n1 x N

        Attributes: These are initialized during the construction of the class object
        -----------
            n1 : int 
                number of boundary conditions at r = 0
            N : int
                number of differential equations
        """
        #shape n1 x N
        dB1dy = np.zeros((self.n1,self.N))
        dB1dy[0][0] = 1.0
        dB1dy[1][1] = 1.0
        dB1dy[2][2] = 1.0
        dB1dy[3][3] = 1.0
        dB1dy[4][4] = 1.0
        return dB1dy

    def dB2dy(self):
        """
        Boundary Conditions at r -> infinity (approximately) .
        Used in relaxation algorithm.
        Shape n2 x N

        Attributes: These are initialized during construction of the class object 
        -----------
            n2: int
               number of boundary conditions at r -> infinity
            N : int
                number of differential equations
        """
        #shape n2 x N
        dB2dy = np.zeros((self.n2,self.N))
        dB2dy[0][1] = 1.0
        dB2dy[1][2] = 1.0
        dB2dy[2][5] = 1.0
        return dB2dy

    def scale(self,k):  
        """
        Scale used in the relaxation algorithm associated with each of the differential equations

        Parameters:
        -----------
            k : int
                Specifies the position of the scale array
        """
        scale = [1.0,0.1,0.1,0.1,0.1,1.0,0.1,0.1]
        return scale[k]

    def findMbs(self,r,y):
        """
        Compute the total mass of the boson star in units of solar mass, 
        the compactness of the star, and galactic scale quantities such as central density.

        Parameters:
        -----------
            r : numpy array
                Radial grid of the solutions to the equations of motions
            y : numpy array 
                Solutions to the equations of motions

        Returns:
        --------
            Cbs : float
                Compactness of the BEC in SI units
            Mbs/msun : float
                Total mass of the BEC in solar mass where msun = mass of the sun
            rc : float
                Core radius of BEC in Natural Units
            rho_c : float
                Core density of BEC in Natural Units 
        """
        A = y[0,...]
        p1 = y[1,...]
        p2 = y[2,...]
        dp1 = y[3,...]
        dp2 = y[4,...]
        B = y[5,...]
        mu1 = y[6,...]
        mu2 = y[7,...]
        lam1 = self.lam1
        lam2 = self.lam2
        lam12 = self.lam12
        mr = self.mr
        mphi = self.mphi
        Mpl = 1.22e19     #in GeV
        c = 3.0e8         #in SI
        G_N = 6.670430e-11    #in SI
        msun = 1.98847e30    #in SI
        f = (1/(2.*B)*(mu1**2*p1**2 + mu2**2*p2**2) + 0.5*(p1**2 + mr**2*p2**2) + 1/(2.*A)*(dp1**2 + dp2**2) + 1/4.*lam1*p1**4 + 1/4.*lam2*p2**4 + 1/4.*lam12*p1**2*p2**2)
        f = interpolate.UnivariateSpline(r,4*np.pi*r**2*f,s=0,k=3)
        I = f.integral(r[0],r[-1])
        def fun(r,I,f,a,b,perc):
            if(r<a): r = a + 0.0001
            if(r>b): r = b - 0.0001
            I90 = f.integral(a,r)
            return I90 - I*perc
        #r90 = optimize.newton(fun,3*r[-1]/4.,args=(I,f,r[0],r[-1],))
        r90 = optimize.brenth(fun,r[0],r[-1],args=(I,f,r[0],r[-1],0.9))
        I90 = f.integral(r[0],r90)
        if not np.allclose(f.integral(r[0],r90),I*0.9) :
            print("R90 not converged :", np.allclose(f.integral(r[0],r90),I*0.9))
        #convert to energy units (GeV)
        Mbs = I/(4.*np.pi*self.G_N*mphi)
        R = r90/mphi
        #convert to SI
        Mbs = Mbs*1.7827e-27
        R = R*0.19733e-15
        #Cbs = G_N*Mbs/(R*c**2) 
        Cbs = G_N/(R*c**2)*I90/(4.*np.pi*self.G_N*mphi)*1.7827e-27
        rho = (1/(2*B)*(mu1**2*p1**2 + mu2**2*p2**2) + 0.5*(p1**2 + mr**2*p2**2) + 1/(2.*A)*(dp1**2 + dp2**2) + 1/4.*lam1*p1**4 + 1/4.*lam2*p2**4 + 1/4.*lam12*p1**2*p2**2)
        rho = rho/(4*np.pi)
        rho_c = rho[0]
        rho = interpolate.UnivariateSpline(r,rho,s=0,k=3)
        try:
            rc = optimize.brenth(lambda r: rho(r) - rho_c/2.,r[0],r[-1])
        except:
            rc = np.nan
        return Cbs,Mbs/msun,rc,rho_c

def initial_guess(phi10,phi20,phi,N,M):
    """
    Initial guess used in the relaxation algorithm. Assumes linear solution 

    Parameters:
    -----------
        phi10 : float
            Central density of the first scalar
        phi20 : float
            Central density of the second scalar
        N : int 
            Number of differential equations
        M : int
            Grid size of solutions

    Returns:
    --------
        r : numpy array
            Radial grid of the solutions to the equations of motions
        y : numpy array 
            Solutions to the equations of motions in the form 
            y[0,...] = A
            y[1,...] = phi1
            y[2,...] = phi2
            y[3,...] = dphi1
            y[4,...] = dphi2
            y[5,...] = B
            y[6,...] = mu1
            y[7,...] = mu2
    """
    A = np.linspace(1.0,1.0,M,endpoint=True)
    B = np.linspace(1.00,1.0,M,endpoint=True)
    phi1 = np.linspace(phi10,phi*1e-6,M,endpoint=True)
    phi2 = np.linspace(phi20,phi*1e-6,M,endpoint=True)
    dphi1 = np.linspace(phi*1e-6,phi*1e-6,M,endpoint=True)
    dphi2 = np.linspace(phi*1e-6,phi*1e-6,M,endpoint=True)
    mu1 = np.linspace(1.0,1.0,M,endpoint=True)
    mu2 = np.linspace(1.0,1.0,M,endpoint=True)
    y = np.zeros((N,M))
    y = np.column_stack(
        (A, phi1, phi2, dphi1, dphi2, B, mu1, mu2)
    )
    return y.T

def interp(x,y,N,M):
    """
    Interpolation helper function

    Parameters:
    -----------
        x : numpy array
            Radial grid of the solutions to the equations of motions
        y : numpy array 
            Solutions to the equations of motions in the form 
        N : int 
            Number of differential equations
        M : int
            New grid size to set the length of the interpolated solutions

    Returns:
    --------
        xnew : numpy arra
            Radial grid of the solutions to the equations of motions
        ynew : numpy array 
            Solutions to the equations of motions in the form 
            y[0,...] = A
            y[1,...] = phi1
            y[2,...] = phi2
            y[3,...] = dphi1
            y[4,...] = dphi2
            y[5,...] = B
            y[6,...] = mu1
            y[7,...] = mu2
    """
    xnew = np.linspace(x[0],x[-1],M)
    ynew = np.zeros((N,M))
    for i in range(N):
        I = interpolate.interp1d(x,y[i,...],kind="cubic")
        ynew[i,...] = I(xnew)
    return xnew,ynew

###############################################
"""
    This function is experimental
"""
def _get_profiles(phi10,phi20,
                 lam1=1,lam2=1,lam12=1,
                 m1=1e-10,m2=1e-10,
                 fr=1e17,
                 rstart=10.0,
                 iteration=100,
                 display=False,
                 tol=10,
                 dx=1.0,
                 max_M=1000,
                 ):
    N = 8
    rout = rstart
    dr = 1.0
    M = int((rstart - 1.0e-5)/dx)
    if phi10 == 0: phi = phi20
    elif phi20 == 0: phi = phi10
    else: phi = min([phi10,phi20])

    rout = rstart
    x = np.linspace(1.0e-5,rout,M)
    y = initial_guess(phi10,phi20,phi,N,M)
    tmpx = np.array(x,copy=False)
    tmpy = np.array(y,copy=False)

    while True:
        res = bosonstar(1.0e-5,rout,M,N,lam1=lam1,lam2=lam2,lam12=lam12,m1=m1,m2=m2,fr=fr,B1=[1.0,phi10,phi20,phi*1e-6,phi*1e-6],B2=[phi*1.0e-6,phi*1.0e-6,1.0],y_guess=y,iteration=iteration,start=True,display=display)
        x = res.x
        y = res.y

        A = y[0,...]
        phi1 = y[1,...]
        phi2 = y[2,...]
        dphi1 = y[3,...]
        dphi2 = y[4,...]
        B = y[5,...]
        mu1 = y[6,...]
        mu2 = y[7,...]

        mask1 = np.where(phi1 < -phi*1e-6)[0]
        mask2 = np.where(phi2 < -phi*1e-6)[0]
        if len(mask1) > 0 or len(mask2) > 0 :
            dr = dr/1.1
            rout = rout - dr 
            M = int((rout - 1.0e-5)/dx)
            if M > max_M: M = max_M
            x = np.linspace(1e-05,rout,M)
            y = initial_guess(phi10,phi20,phi,N,M)
        else:
            dr = dr*1.2
            rout = rout + dr 
            M = int((rout - 1.0e-5)/dx)
            if M > max_M: M = max_M
            x,y = interp(x,y,N,M)
            tmpx = np.array(x,copy=False)
            tmpy = np.array(y,copy=False)
            r1 = abs(dphi1[-1]/dphi1[0])
            r2 = abs(dphi2[-1]/dphi2[0])
            if r1 < tol and r2 < tol:
                break

        #print("M = %g, rout = %g"%(M,rout))
    x,y = interp(x,y,N,len(x)*10)
    return x,y,res

###############################################


def get_profiles(phi10,phi20,
                 lam1=1,lam2=1,lam12=1,
                 m1=1e-10,m2=1e-10,
                 fr=1e17,
                 rstart=10.0,
                 dx=1.0,
                 tol=10,
                 alpha=1.1,
                 max_grid=1000,
                 min_grid=100,
                 iteration=100,
                 display=False):
    """
    This function repeatidly solves the equations of motion using the relaxation algorithm by fist solving
    over the interval (0, rout) and increasing rout until the BEC ground state is found. As rout increases,
    the solutions should converge to the correct asymptotic boundary conditions. Ideally, we want rout to go
    to infinity but we stop at rout such that it approximates the behaviour at r -> infinity. A solution 
    of the scalar profiles with nodes are valid solutions but they correspond to excited states.  This function
    will avoid excited states. 

    Paramters:
    -----------
        phi10: float
            Initial central density of the first scalar
        phi20: float
            Initial central density of the second scalar
        lam1 : float
            Self-coupling of the the first scalar in the equations of motion
        lam2 : float
            Self-coupling of the the second scalar in the equations of motion
        lam12 : float
            Interaction-coupling between the scalars in the equations of motion
        m1 : float
            Mass of the first scalar in units of eV
        m2 : float
            Mass of the second scalar in units of eV
        fr : float
            Decay constant of the scalars in units of GeV. If set to a value, it normalizes the 
            couplings else it can be set to none and not impact the equations of motion.
            Defined to be uniform for all scalars
        rstart: float
            Starting radial position to integrate differential equations up to such that the boundary is 
            (r0, rstart) where r0 is a small number approximating zero. 
        dx : float
            Spacing of the grid
        alpha : float
            Used as a convergance parameter to push rout
        max_grid : float
            Maximum grid length to stop the grid size from adapting 
        min_grid : float
            Minimum grid length so that the grid does not get arbitrarily small
        iteration : float 
            Max number of iterations used in the relaxation algorithm 
        display : bool
            Prints out the boundary values of of the solutions to see if they are converging to the correct
            boundary conditions. 

    Returns:
    --------
        x : numpy array
            Radial grid of the solutions to the equations of motions
        ynew : numpy array 
            The interpolated solutions to the equations of motions with length = 10 * grid size.
            Has the form 
            y[0,...] = A
            y[1,...] = phi1
            y[2,...] = phi2
            y[3,...] = dphi1
            y[4,...] = dphi2
            y[5,...] = B
            y[6,...] = mu1
            y[7,...] = mu2
    """
    N = 8
    #M = int((rstart - 1.0e-5)/dx)
    M = 100
    #if M > 1000: M = 1000
    if M > max_grid: M = max_grid
    if phi10 == 0: phi = phi20
    elif phi20 == 0: phi = phi10
    else: phi = min([phi10,phi20])
    y = initial_guess(phi10,phi20,phi,N,M)
    tmpy = y

    rout = rstart
    x = np.linspace(1.0e-5,rout,M)
    while True:
        res = bosonstar(1.0e-5,rout,M,N,lam1=lam1,lam2=lam2,lam12=lam12,m1=m1,m2=m2,fr=fr,B1=[1.0,phi10,phi20,phi*1e-6,phi*1e-6],B2=[phi*1.0e-6,phi*1.0e-6,1.0],y_guess=y,iteration=iteration,start=True,display=display)
        x = res.x
        y = res.y

        A = y[0,...]
        phi1 = y[1,...]
        phi2 = y[2,...]
        dphi1 = y[3,...]
        dphi2 = y[4,...]
        B = y[5,...]
        mu1 = y[6,...]
        mu2 = y[7,...]

        def update_grid(dx,M,rout):
            M = int((rout - 1.0e-5)/dx)
            if M > max_grid: 
                M = max_grid
                dx = 1.0
            if M < min_grid: 
                M = min_grid
                dx = 1.0
            return dx,M

        mask1 = np.where(phi1 < -phi*1e-6)[0]
        mask2 = np.where(phi2 < -phi*1e-6)[0]
        if len(mask1) > 0 or len(mask2) > 0 :
            rout = rout/1.01
            M = int((rout - 1.0e-5)/dx)
            dx,M = update_grid(dx*0.90,M,rout)
            x = np.linspace(1e-05,rout,M)
            y = initial_guess(phi10,phi20,phi,N,M)
        else:
            rout = rout*alpha
            dx,M = update_grid(dx*1.05,M,rout)
            x,y = interp(x,y,N,M)
            r1 = abs(dphi1[-1]/dphi1[0])
            r2 = abs(dphi2[-1]/dphi2[0])
            if r1 < tol and r2 < tol:
                break

        #print("M = %g"%M)

    x,y = interp(x,y,N,len(x)*10)
    return x,y,res

def get_compactness(x,y,res):
    """
        Returns the mass and compactness of :func: findMbs
    """
    Cbs,Mbs,rc,rho = res.findMbs(x,y)
    return Cbs,Mbs

def get_density(x,y,res):
    """
        Returns the core radius and core density :func: findMbs
    """
    Cbs,Mbs,rc,rho = res.findMbs(x,y)
    return rc,rho

def f(phi10,phi20,**kwargs):
    """
        Helper function to pass to the multiprocessing module. Called by :func: scan below and returns a dictionary 
    """
    try:
        x,y,res = get_profiles(phi10,phi20,**kwargs)
    except:
        return None
    # compute for total star
    C,M,rc,rho = res.findMbs(x,y)
    # compute for only 2 scalar contribution
    _y = np.array(y,copy=True)
    _y[2,:] = 0; _y[4,:] = 0; _y[7,:] = 0
    C1,M1,_,_ = res.findMbs(x,_y)
    # compute for only 1 scalar contribution
    _y = np.array(y,copy=True)
    _y[1,:] = 0; _y[3,:] = 0; _y[6,:] = 0
    C2,M2,_,_ = res.findMbs(x,_y)
    return {"phi1c":y[1,0],"phi2c":y[2,0],
            "x":x,"y":y,
            "Cbs":C,"Mbs":M,
            "M1bs":M1,"C1bs":C1,
            "M2bs":M2,"C2bs":C2,
            "Rc":rc,"Rhoc":rho}

def scan(phis,cores=1,_print=True,split=True,**kwargs):
    """ 
        Returns a dictionary of collected results after using multiprocessing which is 
        called in the module parallel

        Parameters:
        -----------
            phis : numpy array or list 
                Central densities of the scalar fields. 
            cores: int 
                Number of cores to evaluate on
            _print : bool
                To print the central densities during multiprocessing
            split : bool
                Value of True is necessary for the :class: parallel module to properly split the phis
            kwargds : dictionary
                Additional keywords to pass to :func: f

        Dictionary:
        ----------
        phi1c : list
            Initial central densities of the first scalar
        phi12 : list
            Initial central densities of the second scalar
        Cbs : list
            The total compactness of the BEC at the different central densities in SI units
        Mbs : list
            The total mass of the BEC at the different central densities in solar mass units
        M1 : list
            The mass of the first scalar's contribution to the BEC at the different central densities 
            in solar mass units
        M2 : list
            The mass of the second scalar's contribution to the BEC at the different central densities 
            in solar mass units
        Rc : list
            The core radii of the BEC at the different central densities 
            in natural units
        Rhoc : list
            The core densities of the BEC at the different central densities 
            in natural units
    """
    cores = min([len(phis),cores])
    results = {}
    res = parallel(f,cores=cores,_print=True,split=True)(phis,**kwargs)
    for i in range(len(res)):
        if res[i] is None: continue
        for key,val in res[i].items():
            if key not in results:
                results.update({key:[]})
            results[key].append(val)
    return results
    

