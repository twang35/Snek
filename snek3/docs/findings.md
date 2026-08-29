# Findings

**Empty on purpose.** snek3 inherits none of snek2's findings: they are results about snek2's
hyperparameters under TF-Agents, measured on a different collector with a different replay ratio and
a different RNG, and this project has already learned that framework details matter here — its single
largest result came from diffing snek2 against `theSchlong`.

What snek3 does carry is [`invariants.md`](invariants.md), which is a different kind of thing:
properties of the game and the instrumentation rather than conclusions about a knob.

snek2's findings remain readable at `../../snek2/hyperparamTuning/findings.md` and are a reasonable
source of **priors** — which hypotheses are worth testing first — but never of established fact about
snek3.

## Established

*Nothing yet.*

## Falsified

*Nothing yet.*

---

**Format.** One `###` per finding, leading with what is now believed and the measurement that
supports it, then what it replaced. Mark a retraction rather than deleting the section it replaces —
snek2 overturned several of its own findings and each retraction was worth more than the section it
replaced.
