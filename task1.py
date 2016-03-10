import numpy as np

def omegaSearch(A,D):
    K = np.linalg.inv(D).dot(D-A)
    p = max(np.linalg.eigvals(K));
    omega = 2*(1-np.sqrt(1-p**2))/p**2
    return omega;
    


def LUdecomp(A):
    n = len(A)
    B = np.zeros_like(A)
    
    for j in range(0, n-1):
        B[j,j] = 1
        for i in range(j+1, n):
            if A[i,j] != 0.0:
                lower = A[i,j] / A[j,j]
                A[i, j+1:n] = A[i, j+1:n] - lower * A[j, j+1:n]
                A[i,j] = 0
                B[i,j] = lower
    B[n-1,n-1] = 1

    return A, B
    
def LUsolve(A,B,b):
    y = np.linalg.solve(B,b)
    x = np.linalg.solve(A,y)
    
    return x

def lu(A, b):
    sol = []
    
    A,B = LUdecomp(A)
    sol = LUsolve(A,B,b)
    
    return list(sol)

def sor(A, b):
    sol = []
    
    ITERATION_LIMIT = 10
 
    n = len(A)
    D = np.zeros_like(A)
    L = np.zeros_like(A)
    
    for i in range(0,n):
        D[i][i] = A[i][i];
        
    for i in range(0,n):
        for j in range(0,i):
            L[i][j] = -A[i][j];
    
    omega = omegaSearch(A,D)
    if (omega >= 2 or omega <= 0):
        omega = 1
        
    Q = D/omega -L
    K = np.linalg.inv(Q).dot(Q-A)
    c = np.linalg.inv(Q).dot(b)
    x = np.zeros_like(b)
    

    for itr in range(ITERATION_LIMIT):
        x   = K.dot(x) + c;

    sol = x
    
    return list(sol)

def solve(A, b):
    condition = True 
    
    condition = False
    try:
        np.linalg.cholesky(A)
    except np.linalg.linalg.LinAlgError :
        condition = True
    
    if condition:
        print('Solve by lu(A,b)')
        return lu(A,b)
    else:
        print('Solve by sor(A,b)')
        return sor(A,b)
    
    
if __name__ == "__main__":
    ## import checker
    ## checker.test(lu, sor, solve)

    A = np.array([[2.,1,6], [8,3,2], [1,5,1]])
    b = np.array([9., 13, 7]) 
    sol = solve(A,b)
    print(np.round(sol,4))
    
    A = np.array([[6566, -5202, -4040, -5224, 1420, 6229],
         [4104, 7449, -2518, -4588,-8841, 4040],
         [5266,-4008,6803, -4702, 1240, 5060],
         [-9306, 7213,5723, 7961, -1981,-8834],
         [-3782, 3840, 2464, -8389, 9781,-3334],
         [-6903, 5610, 4306, 5548, -1380, 3539.]])
    b = np.array([ 17603,  -63286,   56563,  -26523.5, 103396.5, -27906])
    sol = solve(A,b)
    print(np.round(sol,4))
    
    
    
    
    
    