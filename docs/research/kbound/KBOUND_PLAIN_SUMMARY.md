# K‑Bound — Plain‑Language Summary

**When can a model safely tune itself on new data — and when should it just leave well enough alone?**

## The idea, in plain words

When a model runs into data that doesn't look like what it was trained on, a popular trick is to let it quietly re‑tune itself on that new data — no labels needed. Sometimes this helps a lot. Sometimes it quietly makes things worse. And because there are no labels to check against, the model usually has no way of knowing which is happening.

Most research tries to make the re‑tuning smarter. We asked a simpler, earlier question: **should the model re‑tune itself here at all — and can it figure that out on its own, without labels?**

So we give it three choices, not one: **tune** (adjust to the new data), **freeze** (leave the model as it is), or — when there's genuinely no way to tell — **hold off** and flag it for a person.

## The honest hard part

We prove something uncomfortable but important: there are situations where two completely different realities look *exactly the same* from the outside, yet tuning helps in one and hurts in the other. When that happens, **no label‑free method can ever tell them apart** — not with a cleverer trick, and not with more data. So in those cases, holding off isn't being lazy. It's the only honest thing the model can do.

We also pin down the exact dividing line between "you can tell" and "you can't." In short: a label‑free decision is trustworthy only when the visible disagreement between the old and tuned model is *bigger* than the amount the model's confidence can quietly drift between the old and new data. Above that line, you can decide safely. Below it, you can't. This is also why popular label‑free shortcuts work on easy problems and fall apart on harder shifts — they quietly assume that drift is zero, and it usually isn't.

## The tool

We turn all of this into a practical check you can wrap around *any* existing tuning method. It estimates how much tuning would help, attaches an honest margin of error, and returns one of the three answers — tune, freeze, or hold off — while keeping the "tuned when it shouldn't have" rate below a level you choose. The version for systems that combine several inputs (say, a camera plus a depth sensor) is just this same check applied to a set of fusion options. It's the same idea, not a separate system.

## What we found (the good and the bad, openly)

- On standard image‑corruption stress tests, repeated across several runs, the check reliably does better than both "always tune" and "never tune," and it never once tuned when it shouldn't have.
- On a real hospital‑to‑hospital medical‑imaging shift, once we gave the check a richer set of label‑free signals, it scored a genuine win — decided in advance, tested only once on held‑out data, beating both "always tune" and "never tune," with harmful tuning kept under control.
- On several other real‑world shifts — wildlife cameras, harder ImageNet variants, an office‑objects benchmark, a fraud dataset — the harm was too subtle to spot without labels, so the check did the right thing and held off. We report every one of these, the misses as plainly as the wins.

The pattern *is* the message: it wins exactly where the harm is common and detectable, and it steps back safely everywhere else — which is exactly what the theory says should happen.

## What it can't do

It's not magic. Where tuning is already safe, "always tune" is hard to beat, and the check can only match it. The real‑world win so far is on one dataset, with a modest margin, and it needed both richer signals and a small slice of labeled data to calibrate. And the deepest limit comes straight from the proof: once you're stuck on the wrong side of that dividing line, **more unlabeled data won't rescue you** — we checked, and giving the medical‑imaging case four times more data didn't shrink the uncertainty at all. The only real way through is a little bit of labeled data, not a lot more unlabeled data.

## Why it matters

The lasting contribution isn't a flashy accuracy number — it's **knowing the limits**. We prove where label‑free tuning can be trusted, where it can't, and we give an honest check that tells the two apart and refuses to guess when guessing isn't safe.
