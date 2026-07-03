import sympy as sp
pi, bi, bj, di, dj = sp.symbols('pi b_i b_j delta_i delta_j', real=True)
# d_i(0) = b_i - pi*delta_i ; d_i(1)=b_i+(1-pi)*delta_i so that (1-pi)d_i(0)+pi d_i(1)=b_i. verify:
d_i0 = bi - pi*di; d_i1 = bi + (1-pi)*di
d_j0 = bj - pi*dj; d_j1 = bj + (1-pi)*dj
print("mean check b_i:", sp.simplify((1-pi)*d_i0 + pi*d_i1))   # should be b_i
c_ij = sp.expand((1-pi)*d_i0*d_j0 + pi*d_i1*d_j1)
resid = sp.simplify(c_ij - bi*bj)
print("c_ij - b_i b_j =", resid)
print("factored:", sp.factor(resid))
