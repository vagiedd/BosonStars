from relaxation_method import *
import matplotlib.pyplot as plt

class test(relaxation_method):
    def init(self):
        self.slowc = 1.0

    def f(self,t,y):
        #x = t
        #y[0] = x'
        #y[1] = x
        #y[2] = y
        #f = [2-x, y*x, z]
        x = y.item(0)
        u = y.item(1)
        z = y.item(2)
        f = [z,2.0-x,u*x]
        return np.array(f)

    def dB1dy(self):
        #shape n1 x N
        dB1dy = np.zeros((self.n1,self.N))
        dB1dy[0][0] = 1.0
        dB1dy[1][1] = 1.0
        return dB1dy

    def dB2dy(self):
        #shape n2 x N
        dB2dy = np.zeros((self.n2,self.N))
        dB2dy[0][0] = 1.0
        return dB2dy

    def scale(self,k):  
        return 0.25
        #return 1.

res = test(0,4,1000,3,B1=[1.0,1.0],B2=[1.0],iteration=30,start=True,display=True)
x = res.x
y = res.y
for i in range(res.N):
    plt.plot(x,y[i,...])
plt.show()
