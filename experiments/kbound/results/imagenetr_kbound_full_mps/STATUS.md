# ImageNet-R full sweep — BLOCKED on iWildCam resource contention (flagged)

Relaunch with ALL I/O internal (data ~/kbound_inr, class-index internal, results
~/kbound_inr_results, log internal) verified: process holds NO T9 data files open
(only cwd ref). Despite that, 0/72 conditions after ~14 min, state U, CPU 6-12%.
=> internal-I/O fix necessary but NOT sufficient. Root cause = the iWildCam
extraction (PID 90777, 2h20m, ~203k tiny files on exFAT T9) starving system
I/O + unified memory (~121MB pages free), throttling MPS.

AWAITING USER GO to pause iWildCam (PID 90777). Its archive.tar.gz is fully
downloaded, so extraction re-runs from the archive later (no re-download lost).
