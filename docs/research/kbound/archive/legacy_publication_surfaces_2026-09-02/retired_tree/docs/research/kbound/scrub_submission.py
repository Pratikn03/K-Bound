"""Anonymize + de-ELARA + de-hardware the submission build, add the 3 critical missing
citations, and reposition the multi-candidate novelty note. Operates on kbound_submission.tex,
paper/sections_sub/*.tex, paper/proofs_appendix.tex.
"""
import re, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(HERE, "kbound_submission.tex"),
         os.path.join(HERE, "paper", "proofs_appendix.tex")] + \
        sorted(glob.glob(os.path.join(HERE, "paper", "sections_sub", "*.tex")))

# ordered (regex, replacement) — compounds first
SUBS = [
    # author / date / anonymity (main file)
    (r"\\author\{[^}]*\}", r"\\author{Anonymous Author(s)\\\\ \\small Paper under double-blind review}"),
    (r"\\date\{[^}]*\}", r"\\date{}"),
    # de-anonymizing URLs / handles / repo
    (r"\\url\{[^}]*\}", r"(repository link withheld for double-blind review)"),
    (r"https?://github\.com/[^\s}]+", r"(repository withheld for review)"),
    (r"Pratikn03", r"anon"), (r"Pratik~Niroula", r"Anonymous"), (r"Pratik Niroula", r"Anonymous"),
    (r"AutoML\\_Flagship\\_V8", r"the-repository"), (r"AutoML_Flagship_V8", r"the-repository"),
    (r"Working draft[^.\n]*", r""),
    # ELARA / RGA legacy identity
    (r"ELARA[-/]?U", r"the source system"),
    (r"ELARA\s*/\s*RGA", r"a reliability-gated fusion system"),
    (r"ELARA", r"the source system"), (r"Elara", r"the source system"),
    (r"\bRGA\b", r"reliability-gated attention"),
    # paths that leak the legacy name
    (r"kbound\\_paper/vendored\\_from\\_elara/theory/?", r"the supplementary theory code"),
    (r"vendored\\_from\\_elara", r"vendored\\_components"),
    (r"vendored_from_elara", r"vendored_components"),
    (r"experiments/elara\\_u", r"the score archive"),
    (r"src/elara", r"src/internal"),
    # hardware leaks
    (r"Apple-silicon MPS", r"a single consumer GPU"),
    (r"Apple silicon", r"consumer hardware"), (r"Apple-silicon", r"consumer hardware"),
    (r"M5-GPU", r"a consumer GPU"), (r"M5 GPU", r"a consumer GPU"), (r"Apple M5", r"a consumer GPU"),
    (r"\(MPS\)", r"(GPU)"), (r"\bMPS\b", r"GPU"), (r"\bmps\b", r"gpu"), (r"\bM5\b", r"a consumer GPU"),
]

for fp in files:
    s = open(fp).read(); o = s
    for pat, rep in SUBS:
        s = re.sub(pat, rep, s)
    if s != o:
        open(fp, "w").write(s)

# --- add the 3 critical missing references (after the Anandkumar entry) ---
m = os.path.join(HERE, "kbound_submission.tex"); s = open(m).read()
anchor = r"Tensor Decompositions for Learning Latent Variable Models. Journal of Machine Learning Research 15:2773--2832, 2014."
add = (anchor +
       "\n\\item F.~Parisi, F.~Strino, B.~Nadler, Y.~Kluger. Ranking and Combining Multiple Predictors without Labeled Data. Proceedings of the National Academy of Sciences 111(4):1253--1258, 2014."
       "\n\\item A.~Jaffe, B.~Nadler, Y.~Kluger. Estimating the Accuracies of Multiple Classifiers without Labeled Data. AISTATS 2015."
       "\n\\item M.~Schirmer, M.~Jazbec, C.~A.~Naesseth, E.~Nalisnick. Monitoring Risks in Test-Time Adaptation. arXiv:2507.08721, 2025.")
if "Ranking and Combining Multiple Predictors" not in s:
    s = s.replace(anchor, add, 1); open(m, "w").write(s); print("added 3 citations")

# --- reposition the multi-candidate novelty (credit Parisi/Jaffe-Nadler; differentiate Schirmer-Jazbec) ---
mc = os.path.join(HERE, "paper", "sections_sub", "multicandidate.tex"); t = open(mc).read()
old = r"\paragraph{Weight (honest).} This \emph{enlarges}"
new = (r"\paragraph{Weight (honest).} The agreement estimator itself is classical---the rank-one "
       r"accuracy-from-agreement method of Parisi et al.\ (2014) and Jaffe et al.\ (2015) under "
       r"conditional independence; our contribution is its use as a label-free \emph{decision certificate} "
       r"with the checkable overdetermination diagnostic (Proposition~\ref{prop:cei-test}), not the "
       r"estimator. Unlike reactive risk monitors for TTA (Schirmer \& Jazbec, 2025), the certificate "
       r"decides \emph{before} committing and carries an explicit abstain region. This \emph{enlarges}")
if old in t:
    t = t.replace(old, new, 1); open(mc, "w").write(t); print("repositioned multicandidate novelty note")

print("scrub complete over", len(files), "files")
