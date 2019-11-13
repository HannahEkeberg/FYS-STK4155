import numpy as np




def Franke_Func(x,y, n=20):   #f(x,y) #France function
    term1 = ( 0.75 * np.exp( -((9*x - 2)**2 / 4)  - ((9*y-2)**2 / 4)) )
    term2 = ( 0.75 * np.exp( -((9*x+1)**2 / 49) - ((9*y+1)**2 / 10 )) )
    term3 = ( 0.5 * np.exp( -((9*x-7)**2 / 4 ) - ((9*y-3)**2 / 4)) )
    term4 = -( 0.2 * np.exp( -(9*x-4)**2 - (9*y-7)**2 ) )
    #noise = np.random.normal(0, 0, n)
    return (term1 + term2 + term3 + term4) #+ noise)




def design_matrix(x, y, deg):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((deg + 1)*(deg + 2)/2)
    X = np.ones((N,p))

    for i in range(1, deg + 1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k
    return X
