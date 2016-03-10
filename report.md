UECM3033 Assignment #2 Report
========================================================

- Prepared by: Phoong Jin Zhi
- Tutorial Group: T2

--------------------------------------------------------

## Task 1 --  $LU$ Factorization or SOR method

The reports, codes and supporting documents are to be uploaded to Github at: 

[https://github.com/Jinzhi94/UECM3033_assign2](https://github.com/Jinzhi94/UECM3033_assign2)

Explain your selection criteria here.

To decide which whether LU or SOR method should be use to solve the linear system Ax = b,
we will first check whether the matrix A is positive definite matrix or not. To check this,
we will use Convergence theorem of SOR method, which states that, SOR method will converge if
and only if A is positive definite. Furthermore, by theorem, to check whether a matrix is
positive definite is equivalent to check whether it can apply cholesky factorization.
So if A cant apply cholesky factorization then we use LU method to solve the system.

Explain how you implement your `task1.py` here.

For the LU method, I decompose the matrix A into upper and lower triangular matrix, then solve
them simultaneously to the the solution.
For the SOR method, I set the iteration limit to be 10 so that the solution we get is close enough 
to the actual solution, then i break down the A into two part which are D(diagonal matrix and
L(lower triangular matrix with 0 diagonal).
Before using the formula of SOR method, we will try to seek for the optimal omega value so that the
convergence rate is at maximum, so i implemented the function omegaSearch which will helps us to 
find the greatest eigenvalue of K=D^-1(D-A), then apply the formula omega = 2*(1-np.sqrt(1-p**2))/p**2.
Next we will compute the Q, K and c respectively and finally optained the recursive formula for the system
X = KX + c, iterate it within the iteratio limit we will get the result.

Lastly, i have corrected all the solution we get into 4 decimal places.
---------------------------------------------------------

## Task 2 -- SVD method and image compression

Put here your picture file (Lenna.png)

![Lenna.png](Lenna.png)

How many non zero element in $\Sigma$?

Put here your lower and better resolution pictures. Explain how you generate
these pictures from `task2.py`.

What is a sparse matrix?


-----------------------------------

<sup>last modified: change your date here</sup>
