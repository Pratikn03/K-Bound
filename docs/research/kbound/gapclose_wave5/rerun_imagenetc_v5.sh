#!/bin/bash
# ImageNet-C v5 aggressive arm. Set IC_ROOT to the folder that holds EITHER
#   (A) the 5 tars:  noise.tar blur.tar weather.tar digital.tar extra.tar   (streamed, no extract)
#   (B) flat corruption dirs:  gaussian_noise/<sev>/<wnid>/*.JPEG , motion_blur/<sev>/... etc.
# A category-nested layout ( <root>/noise/gaussian_noise/... ) will NOT be found — flatten it.
set -e
cd /Volumes/T9/uav/AutoML_Flagship_V8
source ~/.venv_wilds/bin/activate
export TORCH_HOME=/Volumes/T9/uav/torch_cache   # pretrained resnet50 f0 cache

# >>> EDIT THIS ONE LINE <<<
IC_ROOT=/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/imagenet-c

echo "IC_ROOT=$IC_ROOT"
echo "contents:"; ls -1 "$IC_ROOT" 2>/dev/null | grep -viE '^\._' | head

# clear any partial/quick-set output so the corrected run starts clean
rm -rf experiments/kbound/results/win_hunt_v5/imagenetc_aggr

# FULL standard-15 ImageNet-C. Deliberate expansion beyond protocol E's 3-noise baseline,
# chosen BEFORE the frozen scoring pass (adds ALL corruptions incl. unfavorable ones -> not
# operating-point shopping). Aggressive operating point (--adapt-lr 0.004, online, batch small);
# arch/severities/compositions per the v5 config. STANDALONE full-benchmark aggressive result
# (NOT a paired benign-vs-E contrast, since E ran only the 3 noise corruptions).
# caffeinate -is keeps the Mac + external drive awake for the whole run (protocol E did this;
# omitting it is what let the T9 sleep mid-run and deadlock the exFAT I/O in uninterruptible wait).
caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --methods tent eata sar --device mps --seed 0 --imagenetc-root "$IC_ROOT" --corruptions gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression --arch resnet50 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --out-results experiments/kbound/results/win_hunt_v5/imagenetc_aggr
