# Comedy or tragedy?

Hausarbeit, Maschinelles Lernen, Digital Humanities, Universität Trier


## The question

Can a classifier tell whether a German play is a comedy or a tragedy purely
from the words its characters speak?

Corpus: [GerDraCor](https://dracor.org/ger), CC0, 409 usable plays from 192
authors, 1500s to 1940s.


## The result

Naive Bayes on spoken text reaches 0.843 where always guessing "comedy"
reaches 0.586. It holds up when tested on playwrights it has never seen.

The accuracy is not the point. The point is what failed to explain it away:

| candidate explanation | effect |
|---|---|
| author identity (66% leakage available) | −0.016 |
| the tokeniser splitting apostrophes | −0.009 |
| dialect comedy | −0.025 |
| chronology | ARI −0.001 against period |
| language change over 400 years | held-out R² below zero |
| verse against prose | real, roughly half |

What survives is social register: court vocabulary against household
vocabulary. That is consistent with the Ständeklausel, though the notebook is
careful to call it an interpretation of a word list rather than a measurement.


## Files

- `analysis.ipynb` — the whole study, eight sections, runs top to bottom
- `drama.py` — downloading, loading, tokenising, splitting
- `results/` — figures and CSV tables (checked in, also regenerated on a run)
- `data/` — corpus cache, created on first run, about 45 MB, not tracked here


## Running it

```
pip install -r requirements.txt
jupyter lab analysis.ipynb
```

Then Restart & Run All. Section 1 downloads the corpus from the DraCor API,
which takes a few minutes once; everything after that reads the cache in
`data/`.


## Two conventions

**Author-disjoint splits.** Whole playwrights go to training or to testing,
never both, so the model cannot win by recognising Kotzebue instead of
recognising comedy. Random splits are reported next to them and the gap is a
result, not a nuisance. The notebook also measures how much leakage a random
split actually offered, because without that number a zero gap would prove
nothing.

**Margin over baseline.** 241 of 409 plays are comedies, so always answering
"comedy" scores 0.586. Several experiments drop plays and move that floor, so
every table carries its own baseline and comparisons use the margin.

The from-scratch Naive Bayes matches scikit-learn's `MultinomialNB` to 0.000
on identical features, which is how I know the hand-written one is right.


## One switch

`LABEL_SOURCE` in `drama.py` picks between DraCor's own `normalizedGenre` and
a label read from the German subtitle. They agree on 99.2% of plays; the
three disagreements are shown in section 1 and are more interesting than the
agreement rate. Flipping it and re-running is the robustness check.


## Corpus version

GerDraCor has no version numbers and keeps growing, so the numbers above are
pinned to a specific commit rather than a play count: `drama.py` records it in
`data/provenance.txt` the first time it runs. The run these results come from
used commit `15a666187e4243368f29c6b7b3f4c461ec94a33c` (committed 2026-08-04).
