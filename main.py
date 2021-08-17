import numpy as np
import matplotlib.pyplot as plt

from BEC.bosonstar import get_profiles, get_compactness, get_density

if __name__ == "__main__":

    mr = 1
    m1 = 1e-10
    m2 = mr*m1

    kwargs = dict(lam1=0,
              lam2=0,
              lam12=0,
              m1=m1,
              m2=m2,
              fr=1e17,
              tol=10,
    )

    x,y,res = get_profiles(1e-3,2e-3,**kwargs)

    print("lambda = %g"%res.lam1)

    C,M = get_compactness(x,y,res)
    rc,rho = get_density(x,y,res)
    print("C = %g"%C)
    print("M = %g"%M)

    plt.xlabel('$r$'); plt.ylabel(r'$A$')
    plt.plot(x,y[0,:])
    plt.tight_layout()
    plt.show()

    plt.xlabel('$r$'); plt.ylabel(r'$\Phi$')
    plt.plot(x,y[1,:],x,y[2,:])
    plt.legend(["1","2"])
    plt.tight_layout()
    plt.show()

    plt.xlabel('$r$'); plt.ylabel(r'$d\Phi / dr$')
    plt.plot(x,y[3,:],x,y[4,:])
    plt.tight_layout()
    plt.show()

    plt.xlabel('$r$'); plt.ylabel(r'$B$')
    plt.plot(x,y[5,:])
    plt.tight_layout()
    plt.show()

    plt.xlabel('$r$'); plt.ylabel(r'$\mu$')
    plt.plot(x,y[6,:],x,y[7,:])
    plt.tight_layout()
    plt.show()
