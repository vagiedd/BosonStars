from relaxation_method import *
import matplotlib.pyplot as plt
class test(relaxation_method):
    def f(self,t,y):
        #y[0] = y
        y = y.item(0)
        f = [y*np.cos(t+y)]
        return np.array(f)

    def jac(self,t,y):
        #y[0] = y
        y = y.item(0)
        jac = [[np.cos(t+y) - y*np.sin(t+y)]]
        return np.array(jac)

    def dB1dy(self):
        #shape n1 x N
        dB1dy = np.zeros((self.n1,self.N))
        dB1dy[0][0] = 1.0
        #dB1dy[1][1] = 1.0
        return dB1dy

    def dB2dy(self):
        #shape n2 x N
        dB2dy = np.zeros((self.n2,self.N))
        #dB2dy[0][0] = 1.0
        return dB2dy

    def scale(self,k):  
        return 0.25
        #return 1.

res = test(0,30,1000,1,B1=[1.0],iteration=300,start=True,display=True)
x = res.x
y = res.y
plt.plot(x,y.T)
plt.show()
#res.plot()
