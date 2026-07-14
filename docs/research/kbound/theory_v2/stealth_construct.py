import numpy as np
import itertools
from scipy.optimize import nnls, linprog

# GOAL: K=4. Construct a JOINT conditional law of correctness C=(C1..C4) in {0,1}^4 given Y
# (we'll use a SINGLE distribution over correctness patterns; by symmetry-of-flip we can fold Y
#  into 'correct/incorrect' and work with the agreement structure directly) such that:
#   - pairwise products c_ij = E[(2C_i-1)(2C_j-1)] form a RANK-ONE matrix bb^T  (=> tau=0, audit/diagnostic blind)
#   - BUT the TRUE advantages b_i^true = E[2C_i-1] are DIFFERENT from the b recovered by the rank-1 fit.
# i.e. correctness indicators are NOT independent (CEI false) yet pairwise agreements are exactly H-realizable.
#
# Approach: pick a target rank-1 c = bb^T with some b_fit. The product-ratio estimator returns b_fit.
# Now find a joint law over {0,1}^4 (patterns of correctness 1[f_i=Y]) whose
#   * pairwise second moments give exactly c_ij = b_fit_i b_fit_j  (i!=j), AND
#   * first moments (=> true b) are something ELSE: b_true != b_fit.
# Note E[2C_i-1] = b_true_i is a FIRST moment; c_ij is a (centered-ish) SECOND cross moment.
# In a joint law these are partly independent: we have 2^4=16 free pattern-probs (sum=1),
# constrained by 6 pairwise-second-moment equations + nonneg. First moments are then free-ish.
# Let's solve a feasibility LP: variables p (16,), constraints:
#   sum p = 1; p>=0
#   for each pair (i,j): sum_patterns p * (2c_i-1)(2c_j-1) = c_ij_target
# Then MAXIMIZE/inspect the resulting first moments b_true; show they differ from b_fit.

K = 4
patterns = np.array(list(itertools.product([0,1], repeat=K)))  # 16 x 4, entries = correctness C_i
S = 2*patterns - 1                                             # signs s_i = 2C_i-1 in {-1,+1}

# choose b_fit (the rank-1 'recovered' advantages) with good margins
b_fit = np.array([0.6, 0.55, 0.5, 0.45])
c_target = np.outer(b_fit, b_fit)

# build pairwise second-moment constraint matrix
pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
A_eq = [np.ones(16)]                       # sum p = 1
b_eq = [1.0]
for (i,j) in pairs:
    A_eq.append(S[:,i]*S[:,j])             # E[s_i s_j] = c_ij
    b_eq.append(c_target[i,j])
A_eq = np.array(A_eq); b_eq = np.array(b_eq)

# We want b_true (first moments E[s_i]) to differ from b_fit. Maximize b_true[0]-b_fit[0] via LP objective.
# objective: minimize -(E[s_0]) = -(sum p s_0)
c_obj = -(S[:,0])
res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=[(0,1)]*16, method="highs")
print("LP success:", res.success, res.message)
p = res.x
b_true = S.T @ p
# verify pairwise products of the REALIZED law
c_real = np.zeros((K,K))
for i in range(K):
    for j in range(K):
        c_real[i,j] = np.sum(p*S[:,i]*S[:,j])
od = ~np.eye(K,dtype=bool)
print("b_fit (rank-1 recovered):", b_fit)
print("b_true (this law's first moments):", b_true.round(4))
print("max |b_true - b_fit| :", round(np.max(np.abs(b_true-b_fit)),4), " <-- if >0: identification BROKEN with tau=0")
print("max offdiag |c_real - c_target(rank1)| :", np.format_float_scientific(np.max(np.abs((c_real-c_target)[od])),precision=2))
# 2x2 minors (the tau statistic): should all vanish since c_real == rank-1 c_target
minors = [c_real[0,1]*c_real[2,3]-c_real[0,2]*c_real[1,3],
          c_real[0,1]*c_real[2,3]-c_real[0,3]*c_real[1,2],
          c_real[0,2]*c_real[1,3]-c_real[0,3]*c_real[1,2]]
print("tau minors (should be ~0):", [round(m,6) for m in minors])
print("tau = max-min of products:", round(max([c_real[0,1]*c_real[2,3],c_real[0,2]*c_real[1,3],c_real[0,3]*c_real[1,2]])
                                          -min([c_real[0,1]*c_real[2,3],c_real[0,2]*c_real[1,3],c_real[0,3]*c_real[1,2]]),6))
# is CEI violated? check independence: under CEI joint p would factor. Test a triple moment vs product.
trip_real = np.sum(p*S[:,0]*S[:,1]*S[:,2])
trip_ind = b_true[0]*b_true[1]*b_true[2]
print("E[s0 s1 s2] realized:", round(trip_real,4), " vs product of means:", round(trip_ind,4),
      " -> CEI", "VIOLATED" if abs(trip_real-trip_ind)>1e-3 else "ok")
np.save("/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/stealth_p.npy", p)
np.save("/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/stealth_meta.npy",
        {"b_fit":b_fit,"b_true":b_true,"patterns":patterns}, allow_pickle=True)
