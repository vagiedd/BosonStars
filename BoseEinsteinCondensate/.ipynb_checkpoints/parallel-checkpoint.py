import numpy as np
class parallel:
    def __init__(self,f,cores=8,split=False,_print=False):
        self.f = f
        self.split = split
        self.cores = cores
        self._print = _print
    def func(self,x):
        if self.split :
            f = self.f(*x,*self.args,**self.kwargs)
        else :
            f = self.f(x,*self.args,**self.kwargs)
        if self._print : 
            sol = np.array([x])
            #sol = np.append(sol,f).ravel()
            #print(*(tuple(sol)))
            print(np.array(x))
        return f
    def __call__(self,x,*args,**kwargs):
        from multiprocessing import get_context,Pool
        self.args = args
        self.kwargs = kwargs
        with get_context("spawn").Pool(self.cores) as pool:
            y = pool.map_async(self.func,x).get()
            pool.close()
            pool.join()
        return y
