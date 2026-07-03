import numpy as np
rng = np.random.default_rng(0)

def simulate(pi, qmat, n=4_000_000, seed=0):
    rng = np.random.default_rng(seed)
    K = qmat.shape[0]
    Y = (rng.random(n) < pi).astype(int)
    q = qmat[:, Y]
    C = (rng.random((K, n)) < q).astype(int)
    F = np.where(C == 1, Y[None, :], 1 - Y[None, :])
    a = C.mean(axis=1)
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = (F[i] == F[j]).mean()
    return a, A

def formula_naive(a):
    return np.outer(a, a) + np.outer(1-a, 1-a)

def formula_correct(pi, qmat):
    K = qmat.shape[0]; A = np.zeros((K, K))
    for y, py in [(0, 1-pi), (1, pi)]:
        qy = qmat[:, y]
        A += py * (np.outer(qy, qy) + np.outer(1-qy, 1-qy))
    return A

K=4
od = ~np.eye(K,dtype=bool)

print("CASE 1 (SYMMETRIC q_i(0)=q_i(1)) -- OFF-DIAGONAL ONLY:")
qsym = rng.uniform(0.6, 0.9, size=K)
qmat = np.stack([qsym, qsym], axis=1)
a, A = simulate(0.5, qmat, seed=1)
naive = formula_naive(a)
print("  a:", np.round(a,4))
print("  max OFFDIAG |A_emp - naive| =", np.format_float_scientific(np.max(np.abs((A-naive)[od])),precision=3))
b=2*a-1; c_emp=2*A-1
print("  max OFFDIAG |c_ij - b_i b_j| =", np.format_float_scientific(np.max(np.abs((c_emp-np.outer(b,b))[od])),precision=3))

print()
print("CASE 1b (SYMMETRIC, but UNBALANCED classes pi=0.25):")
a, A = simulate(0.25, qmat, seed=11)
naive = formula_naive(a)
print("  a:", np.round(a,4))
print("  max OFFDIAG |A_emp - naive| =", np.format_float_scientific(np.max(np.abs((A-naive)[od])),precision=3),
      "  <-- symmetric+unbalanced: does it still hold?")
b=2*a-1; c_emp=2*A-1
print("  max OFFDIAG |c_ij - b_i b_j| =", np.format_float_scientific(np.max(np.abs((c_emp-np.outer(b,b))[od])),precision=3))

print()
print("CASE 2 (ASYMMETRIC q_i(0)!=q_i(1), pi=0.5) -- OFF-DIAGONAL ONLY:")
q0 = rng.uniform(0.5, 0.95, size=K); q1 = rng.uniform(0.5, 0.95, size=K)
qmat = np.stack([q0, q1], axis=1)
a, A = simulate(0.5, qmat, seed=2)
naive = formula_naive(a); correct = formula_correct(0.5, qmat)
print("  q0:",np.round(q0,3)," q1:",np.round(q1,3))
print("  a:", np.round(a,4))
print("  max OFFDIAG |A_emp - naive(a)| =", np.format_float_scientific(np.max(np.abs((A-naive)[od])),precision=3),
      "  <-- paper's identity")
print("  max OFFDIAG |A_emp - correctCEI| =", np.format_float_scientific(np.max(np.abs((A-correct)[od])),precision=3),
      "  <-- true CEI formula")
b=2*a-1; c_emp=2*A-1
print("  max OFFDIAG |c_ij - b_i b_j| =", np.format_float_scientific(np.max(np.abs((c_emp-np.outer(b,b))[od])),precision=3),
      "  <-- c_ij=b_i b_j claim")

# Now: what IS the correct invariant under plain CEI? Define generalized advantage per class.
# c_ij = 2 A_ij -1 = E_Y[(2C_i-1)(2C_j-1)] = E_Y[ d_i(Y) d_j(Y) ] where d_i(y)=2q_i(y)-1.
# = (1-pi) d_i(0)d_j(0) + pi d_i(1)d_j(1).  This is rank<=2, NOT rank-1 in general.
print()
print("CASE 2 -- correct invariant: c_ij = (1-pi)d_i(0)d_j(0)+pi d_i(1)d_j(1), rank<=2:")
d0=2*q0-1; d1=2*q1-1
c_theory = 0.5*np.outer(d0,d0)+0.5*np.outer(d1,d1)
print("  max OFFDIAG |c_emp - c_theory(rank2)| =", np.format_float_scientific(np.max(np.abs((c_emp-c_theory)[od])),precision=3))
# rank of c (offdiag structure): check via the 2x2 minor that rank-1 would force to vanish
m = c_emp.copy(); np.fill_diagonal(m, np.nan)
minor = c_emp[0,1]*c_emp[2,3]-c_emp[0,2]*c_emp[1,3]
print("  2x2 minor c01*c23 - c02*c13 =", np.round(minor,5), "(=0 iff rank-1; nonzero => asymmetry breaks rank-1)")
