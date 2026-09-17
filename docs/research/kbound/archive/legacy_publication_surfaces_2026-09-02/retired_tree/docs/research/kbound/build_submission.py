"""Transform the submission copy: move every proof to a Deferred-Proofs appendix,
leaving a titled pointer in the body. Operates on kbound_submission.tex (inline proofs)
and paper/sections_sub/*.tex. Writes paper/proofs_appendix.tex. Idempotent-ish: run once
on the fresh copies.
"""
import re, os

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "kbound_submission.tex")
SECDIR = os.path.join(HERE, "paper", "sections_sub")
APPOUT = os.path.join(HERE, "paper", "proofs_appendix.tex")

PROOF = re.compile(r"\\begin\{proof\}(\[[^\]]*\])?(.*?)\\end\{proof\}", re.DOTALL)
KINDS = ("theorem", "proposition", "lemma", "corollary")
POINTER = r"\emph{(Proof deferred to Appendix~\ref{app:proofs}.)}"

collected = []   # ordered list of appendix proof entries

def kind_label(prefix):
    labs = re.findall(r"\\label\{([^}]+)\}", prefix)
    label = labs[-1] if labs else None
    kinds = re.findall(r"\\begin\{(%s)\}" % "|".join(KINDS), prefix)
    kind = kinds[-1].capitalize() if kinds else "Result"
    return kind, label

def transform(text):
    def repl(m):
        prefix = text[:m.start()]
        kind, label = kind_label(prefix)
        body = m.group(2)
        head = "Proof of %s~\\ref{%s}" % (kind, label) if label else "Proof of %s" % kind
        collected.append("\\begin{proof}[%s]%s\\end{proof}\n" % (head, body))
        return POINTER
    return PROOF.sub(repl, text)

# 1) main file inline proofs (process first for ordering)
s = open(MAIN).read()
s2 = transform(s)
open(MAIN, "w").write(s2)
n_main = s.count(r"\begin{proof}")

# 2) section files in INPUT order as they appear in the main file
order = re.findall(r"\\input\{paper/sections_sub/([A-Za-z0-9_]+)\}", s2)
n_sec = 0
for name in order:
    fp = os.path.join(SECDIR, name + ".tex")
    if not os.path.exists(fp):
        continue
    t = open(fp).read()
    n_sec += t.count(r"\begin{proof}")
    open(fp, "w").write(transform(t))

# 3) write the appendix collection
hdr = ("% Auto-generated deferred proofs (build_submission.py). Do not edit by hand.\n"
       "% Each proof is titled by the result it proves; labels resolve to the body statements.\n\n")
open(APPOUT, "w").write(hdr + "\n".join(collected))

# 4) insert the deferred-proofs appendix right after \appendix in the main file
s3 = open(MAIN).read()
ins = ("\\appendix\n\\section{Deferred proofs}\\label{app:proofs}\n"
       "All proofs of the results stated in the body are collected here.\n"
       "\\input{paper/proofs_appendix}\n\n")
if "\\section{Deferred proofs}" not in s3:
    s3 = s3.replace("\\appendix\n", ins, 1)
    open(MAIN, "w").write(s3)

print("inline proofs moved (main): %d | section proofs moved: %d | total collected: %d"
      % (n_main, n_sec, len(collected)))
print("pointers in main now:", open(MAIN).read().count("Proof deferred to Appendix"))
