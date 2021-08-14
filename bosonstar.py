from relaxation_method import *
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np

class bosonstar(relaxation_method):
    def init(self,lam1=1,lam2=1,lam12=1,m1=1e-10,m2=1e-10,fr=1e17):
        self.slowc = 1.0

        self.mphi = m1*1.e-9
        self.mr = m2/m1

        self.fr = fr # in GeV
        self.G_N = 6.70883e-39
        if fr is None:
            self.lam1 = lam1
            self.lam2 = lam2
            self.lam12 = lam12
        else:
            self.lam1 = lam1/(4*np.pi*self.G_N*self.fr**2) 
            self.lam2 = lam2/(4*np.pi*self.G_N*self.fr**2) 
            self.lam12 = lam12/(4*np.pi*self.G_N*self.fr**2) 
        self.c = np.sign(self.mr)

        self.tol = 1.0e-8

    def f(self,r,y):
        #x = t
        #y[0] = A
        #y[1] = phi1
        #y[2] = phi2
        #y[3] = dphi1
        #y[4] = dphi2
        #y[5] = B 
        #y[6] = mu1 
        #y[7] = mu1 
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
        f3 =-((1/(2*B*r))*(2*B*dphi1 + 2*A*B*dphi1 - 2*A*B*c*phi1*r + 2*A*mu1**2*phi1*r - \
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
        #x = t
        #y[0] = A
        #y[1] = phi1
        #y[2] = phi2
        #y[3] = dphi1
        #y[4] = dphi2
        #y[5] = B 
        #y[6] = mu1 
        #y[7] = mu1 
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
        #shape n1 x N
        dB1dy = np.zeros((self.n1,self.N))
        dB1dy[0][0] = 1.0
        dB1dy[1][1] = 1.0
        dB1dy[2][2] = 1.0
        dB1dy[3][3] = 1.0
        dB1dy[4][4] = 1.0
        return dB1dy

    def dB2dy(self):
        #shape n2 x N
        dB2dy = np.zeros((self.n2,self.N))
        dB2dy[0][1] = 1.0
        dB2dy[1][2] = 1.0
        dB2dy[2][5] = 1.0
        return dB2dy

    def scale(self,k):  
        scale = [1.0,0.1,0.1,0.1,0.1,1.0,0.1,0.1]
        return scale[k]

    def findMbs(self,r,y):
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
        f = (1/(2*B)*(mu1**2*p1**2 + mu2**2*p2**2) + 0.5*(p1**2 + mr**2*p2**2) + 1/(2.*A)*(dp1**2 + dp2**2) + 1/4.*lam1*p1**4 + 1/4.*lam2*p2**4 + 1/4.*lam12*p1**2*p2**2)
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

    def findM2(self,r,y):
        A = y[0,...]
        p1 = 0*y[1,...]
        p2 = y[2,...]
        dp1 = 0*y[3,...]
        dp2 = y[4,...]
        B = y[5,...]
        mu1 = 0*y[6,...]
        mu2 = y[7,...]
        lam1 = 0*self.lam1
        lam2 = self.lam2
        lam12 = 0*self.lam12
        mr = self.mr
        mphi = self.mphi
        Mpl = 1.22e19     #in GeV
        c = 3.0e8         #in SImu1
        G_N = 6.670430e-11    #in SI
        msun = 1.98847e30    #in SI
        f = 4*np.pi*r**2*(1/(2*B)*(mu1**2*p1**2 + mu2**2*p2**2) + 0.5*(p1**2 + mr**2*p2**2) + 1/(2.*A)*(dp1**2 + dp2**2) + 1/4.*lam1*p1**4 + 1/4.*lam2*p2**4 + 1/4.*lam12*p1**2*p2**2)
        f = interpolate.UnivariateSpline(r,f,s=0,ext=2)
        I = f.integral(r[0],r[-1])
        def fun(r,I,f,a,b):
            if(r<a): r = a + 0.0001
            if(r>b): r = b - 0.0001
            I90 = f.integral(a,r)
            return I90 - I*0.9
        #r90 = optimize.newton(fun,3*r[-1]/4.,args=(I,f,r[0],r[-1],))
        r90 = optimize.brenth(fun,r[0],r[-1],args=(I,f,r[0],r[-1],))
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
        return Cbs,Mbs/msun

def check_converge(r,phi,dphi,mu,phi0,tol=1.0e-18):
    f = phi*(np.sqrt(np.abs(1.0-mu**2)) + 1/r**2) + dphi**2
    if(np.abs(f)<=tol): return True
    else: return False

def initial_guess(phi10,phi20,phi,N,M):
    A = np.linspace(1.0,1.0,M,endpoint=True)
    B = np.linspace(1.00,1.0,M,endpoint=True)
    phi1 = np.linspace(phi10,phi*1e-6,M,endpoint=True)
    phi2 = np.linspace(phi20,phi*1e-6,M,endpoint=True)
    dphi1 = np.linspace(phi*1e-6,phi*1e-6,M,endpoint=True)
    dphi2 = np.linspace(phi*1e-6,phi*1e-6,M,endpoint=True)
    mu1 = np.linspace(1.0,1.0,M,endpoint=True)
    mu2 = np.linspace(1.0,1.0,M,endpoint=True)
    y = np.zeros((N,M))
    y[0,...] = A
    y[1,...] = phi1
    y[2,...] = phi2
    y[3,...] = dphi1
    y[4,...] = dphi2
    y[5,...] = B
    y[6,...] = mu1
    y[7,...] = mu2
    return y

def interp(x,y,N,M):
    xnew = np.linspace(x[0],x[-1],M)
    ynew = np.zeros((N,M))
    for i in range(N):
        I = interpolate.interp1d(x,y[i,...],kind="cubic")
        ynew[i,...] = I(xnew)
    return xnew,ynew

def get_profiles(phi10,phi20,lam1=1,lam2=1,lam12=1,m1=1e-10,m2=1e-10,fr=1e17,rstart=10.0,dx=1.0,display=False,iteration=100):
    N = 8
    M = int((rstart - 1.0e-5)/dx)
    if M > 1000: M = 1000
    if phi10 == 0: phi = phi20
    elif phi20 == 0: phi = phi10
    else: phi = min([phi10,phi20])
    y = initial_guess(phi10,phi20,phi,N,M)
    tmpy = y

    rout = rstart
    converge1 = False
    converge2 = False
    while True:
        res = bosonstar(1.0e-5,rout,M,N,lam1=lam1,lam2=lam2,lam12=lam12,m1=m1,m2=m2,fr=fr,B1=[1.0,phi10,phi20,phi*1e-6,phi*1e-6],B2=[phi*1.0e-6,phi*1.0e-6,1.0],y_guess=y,iteration=100,start=True,display=display)
        x = res.x
        y = res.y
        phi1 = y[1,...]
        phi2 = y[2,...]
        dphi1 = y[3,...]
        dphi2 = y[4,...]
        mu1 = y[6,...]
        mu2 = y[7,...]
        tol = 1.0e-6
        mask1 = np.where(phi1 < 0)[0]
        mask2 = np.where(phi2 < 0)[0]
        mask3 = np.where(dphi1 > 0)[0]
        mask4 = np.where(dphi2 > 0)[0]
        if len(mask1) > 0 or len(mask2) > 0 :
            rout = rout/1.01
            M = int((rout - 1.0e-5)/dx)
            if M > 1000: M = 1000
            y = initial_guess(phi10,phi20,phi,N,M)
        else:
            rout = rout*1.1
            M = int((rout - 1.0e-5)/dx)
            if M > 1000: M = 1000
            x,y = interp(x,y,N,M)
            if dphi1[-1] > 1.0e-5*dphi1[0] and dphi2[-1] > 1.0e-5*dphi2[0] :
                break
    x,y = interp(x,y,N,10000)
    return x,y,res

def get_compactness(x,y,res):
    Cbs,Mbs,rc,rho = res.findMbs(x,y)
    return Cbs,Mbs

def get_density(x,y,res):
    Cbs,Mbs,rc,rho = res.findMbs(x,y)
    return rc,rho

def f(phi10,phi20,**kwargs):
    x,y,res = get_profiles(phi10,phi20,**kwargs)
    C,M,rc,rho = res.findMbs(x,y)
    C2,M2 = res.findM2(x,y)
    return {"phi1c":y[1,0],"phi2c":y[2,0],"Cbs":C,"Mbs":M,"M2bs":M2,"C2bs":C2,"Rc":rc,"Rhoc":rho}

def scan(phis,cores=1,parallel={"_print":True,"split":True},**kwargs):
    import parallel as par
    cores = min([len(phis),cores])
    results = {"phi1c":[],"phi2c":[],"Cbs":[],"Mbs":[],"M2bs":[],"C2bs":[],"Rc":[],"Rhoc":[]}
    res = par.parallel(f,cores=cores,**parallel)(phis,**kwargs)
    for i in range(len(res)):
        for key,val in res[i].items():
            results[key].append(val)
    return results
    

if __name__ == "__main__":
    kwargs = dict(lam1=1,lam2=1,lam12=1,m1=1e-10,m2=1e-10,fr=1e17)
    x,y,res = get_profiles(6e-3,0,**kwargs)
    plt.plot(x,y[1,:],x,y[2,:])
    plt.show()

    cores = 1

    phi1 = 10**(np.linspace(np.log10(1e-8),np.log10(8e-2),50))
    phi2 = np.zeros_like(phi1)
    phis = np.column_stack((phi1,phi2))
    results = scan(phis,cores=cores,**kwargs)
    kwinterp = {"kind":"cubic","bounds_error":False,"fill_value":(np.nan,)}
    C = interpolate.interp1d(results["phi1c"],results["Cbs"],**kwinterp)
    M = interpolate.interp1d(results["phi1c"],results["Mbs"],**kwinterp)
    phi = np.linspace(phi1[0],phi1[-1],len(phi2)*100)

    plt.xlabel('$C_{BS}$');plt.ylabel('$M_{BS}$')
    plt.plot(C(phi),M(phi))
    #plt.xscale('log')
    #plt.yscale('log')
    with open("cvm-0.txt","rb") as f:
        x,y = np.loadtxt(f).T
    plt.plot(x,y,color="magenta")
    plt.show()
