import numpy as np
rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# CEI HYPOTHESIS CHECK
# Setting: Y in {0,1} on region D. Predictors f_i. Correctness C_i = 1[f_i = Y].
# Paper claims: under "C_i mutually independent given Y" (Def CEI),
#   A_ij := P(f_i = f_j) = a_i a_j + (1-a_i)(1-a_j),  a_i = P(f_i=Y).
# We test whether plain CEI (conditional independence given Y) suffices,
# or whether per-class symmetric accuracy P(C_i=1|Y=0)=P(C_i=1|Y=1) is needed.
#
# KEY SUBTLETY: f_i = f_j is NOT equivalent to (C_i = C_j) in general!
#   In BINARY: f_i = f_j  <=>  C_i = C_j  (both right => both =Y; both wrong => both = 1-Y, equal).
#   This IS true in binary. So A_ij = P(C_i = C_j).
# Under CEI: P(C_i=C_j) = E_Y[ P(C_i=C_j|Y) ] = E_Y[ q_i(Y)q_j(Y) + (1-q_i(Y))(1-q_j(Y)) ]
#   where q_i(y) = P(C_i=1|Y=y).
# This equals a_i a_j + (1-a_i)(1-a_j) ONLY when q_i(y) does not depend on y (per-class symmetry),
#   OR by an algebraic coincidence. Let's test.
# ---------------------------------------------------------------

def simulate(pi, qmat, n=4_000_000, seed=0):
    """pi=P(Y=1); qmat[i,y]=P(C_i=1|Y=y). Returns empirical a_i, A_ij, and the two formulas."""
    rng = np.random.default_rng(seed)
    K = qmat.shape[0]
    Y = (rng.random(n) < pi).astype(int)
    q = qmat[:, Y]                      # K x n, prob correct per sample
    C = (rng.random((K, n)) < q).astype(int)   # correctness, independent across i given Y (CEI holds by construction)
    # predictions: f_i = Y if correct else 1-Y
    F = np.where(C == 1, Y[None, :], 1 - Y[None, :])
    a = C.mean(axis=1)                  # P(f_i=Y)
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = (F[i] == F[j]).mean()
    return a, A

def formula_naive(a):
    """a_i a_j + (1-a_i)(1-a_j) -- paper's claimed A_ij"""
    return np.outer(a, a) + np.outer(1-a, 1-a)

def formula_correct(pi, qmat):
    """True E_Y[ q_i(Y)q_j(Y)+(1-q_i(Y))(1-q_j(Y)) ] under CEI"""
    K = qmat.shape[0]
    A = np.zeros((K, K))
    for y, py in [(0, 1-pi), (1, pi)]:
        qy = qmat[:, y]
        A += py * (np.outer(qy, qy) + np.outer(1-qy, 1-qy))
    return A

print("="*70)
print("CASE 1: per-class SYMMETRIC accuracies q_i(0)=q_i(1)  (the corrected H)")
print("="*70)
K=4
qsym = rng.uniform(0.6, 0.9, size=K)
qmat = np.stack([qsym, qsym], axis=1)   # symmetric: same accuracy each class
pi = 0.5
a, A = simulate(pi, qmat, seed=1)
naive = formula_naive(a)
print("a (empirical):", np.round(a,4))
print("max|A_emp - naive(a a + (1-a)(1-a))| =", np.round(np.max(np.abs(A-naive)),5))
# also check c_ij = b_i b_j with b=2a-1
b = 2*a-1
c_emp = 2*A-1
print("max|c_ij - b_i b_j| =", np.round(np.max(np.abs(c_emp - np.outer(b,b))),5), "(off-diag relevant)")

print()
print("="*70)
print("CASE 2: per-class ASYMMETRIC accuracies q_i(0) != q_i(1), pi=0.5")
print("  -- does plain CEI still give A_ij = a_i a_j + (1-a_i)(1-a_j)?")
print("="*70)
q0 = rng.uniform(0.5, 0.95, size=K)
q1 = rng.uniform(0.5, 0.95, size=K)
qmat = np.stack([q0, q1], axis=1)
pi=0.5
a, A = simulate(pi, qmat, seed=2)
naive = formula_naive(a)
correct = formula_correct(pi, qmat)
print("q0:", np.round(q0,3), " q1:", np.round(q1,3))
print("a (empirical):", np.round(a,4))
print("max|A_emp - naive| =", np.round(np.max(np.abs(A-naive)),5), " <-- if >0, paper's identity FAILS under plain CEI")
print("max|A_emp - correct CEI formula| =", np.round(np.max(np.abs(A-correct)),6))
b = 2*a-1
c_emp = 2*A-1
od = ~np.eye(K,dtype=bool)
print("max off-diag |c_ij - b_i b_j| =", np.round(np.max(np.abs((c_emp - np.outer(b,b))[od])),5),
      "<-- if >0, c_ij = b_i b_j FAILS")

print()
print("="*70)
print("CASE 3: ASYMMETRIC accuracies, pi=0.5, balanced classes -- special check")
print("  If pi=0.5 AND q_i(0)+q_i(1) symmetric in a particular way?")
print("="*70)
# Try q_i(1) = 1 - q_i(0) (a structured asymmetry: 'flip' style)
q0 = rng.uniform(0.55, 0.9, size=K)
q1 = q0.copy()  # start symmetric then perturb one
q1[0] = 0.4  # make model 0 asymmetric
qmat = np.stack([q0,q1],axis=1)
a, A = simulate(0.5, qmat, seed=3)
b=2*a-1; c_emp=2*A-1
print("q0:",np.round(q0,3)," q1:",np.round(q1,3))
print("a:",np.round(a,4))
print("max off-diag |c_ij - b_i b_j| =", np.round(np.max(np.abs((c_emp-np.outer(b,b))[od])),5))
