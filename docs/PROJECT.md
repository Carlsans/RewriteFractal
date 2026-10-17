# RewriteFractal: how it works, what worked, what did not

Written from the code and from the output left on disk: about 29,700 saved genomes (`images/*.json`) and
59,900 PNGs produced between 2025-11-30 and 2025-12-20, plus earlier snapshots. The Optuna studies lived in a MySQL
database that was not available, so hyperparameter results are not summarised here. Statements marked *(inferred)*
are conclusions from the evidence, not recorded results.

## The idea

A **genome** has three parts:

1. a **start string**, for example `C*BdbZZa`;
2. **rewrite rules**, an ordered mapping such as `{'aEh': 'imrF', 'bI': 'MHd', ...}` applied to the whole string in
   parallel, up to 100 times;
3. **constants**: a value for every capital letter (a complex number), a function for every lower-case letter, and
   the viewing window (`center`, `range`).

Every string produced by the rewriting is one **formula**, read left to right. `Z` is the running value and `C` is the pixel.
A capital that is not `Z` or `C` is a complex constant; a lower-case letter is a function applied to what follows; `+ - * /` join
terms. The `n`-th line of the rewriting is the update rule for the `n`-th iteration, so the rewriting system defines a
**different formula at each iteration** rather than one fixed map. The pixel's escape iteration, and the value it escaped
with, give the picture.

Function constants (lower case) are one of: `Taylorian` (a coefficient vector), `Polynomial`, `Rational`, `Power`,
`Trigonometric` (sin cos tan cosec sec cotan), `Exponential`, `Logarithmic`.

## Pipeline

| Stage | File | Notes |
|---|---|---|
| Random genome | `RewriterRandomGenerator.py` | Start string 3-15 symbols, 2-35 rules of 1-5 symbols, random constants and window. |
| Rewrite | `RewriterEngine.py` (`rewrite`) | Rules applied via placeholders so replacements do not cascade within one step. Stops if a line exceeds `maxlinelenght` (200 in evaluation, 20000 when drawing). A Cython copy (`.pyx`) existed for speed. |
| Sanity | `RewriterEngine.sanitycheck` | At most 40 functions and 20 `Z`/`C` per line. |
| Run | `RewriterEngine.calculateConstants`, `runAllLines`, `Fractal.py` | Complex128 tensors on CUDA, escape test `Re(Z)+Im(Z) > 4`. Early exit if the first line yields a zero `Z` or the bailout is still trivial at line 5. |
| Colour | `ComplexMatrix.py`, `TimeoftheDayColormap.py` | Exit values are replaced by their **rank**, then real rank times imaginary rank. The colormap changed with the hour of the day. |
| Fitness | `GeneticAlgorithm.evaluate`, `Sieve.py`, `ClaudesMethod.py` | See below. |
| Search | `GeneticAlgorithm.main`, `StudyManager.py`, `GAHyperParameters.py` | DEAP population, custom crossover and mutation, Optuna tunes the GA's own parameters. |
| Archive | `draw_individual`, `PoolManager.py` | Each good individual is stored as `.json` (jsonpickle), a `.txt` of its lines and a PNG. |
| Ownership | `Website/WebsiteScripts.py` | Downloads images and timestamps them with OpenTimestamps (see the `fractalownership` repository). |

## The fitness the last runs used

Evaluation is cheap and staged: rewrite, sanity-check, render at 64x64, reject if the bailout is flat. The score that
the December 2025 runs optimised was, in `GeneticAlgorithm.evaluate`:

```
entropy = bailout_png_size * 3  -  elapsed_time * 3         # each min-max normalised over the run
fitness = -entropy_normalised * 100000                       # DEAP minimises
```

`bailout_png_size` is the size of the PNG of the bailout image divided by the pixel count: a compression-based
estimate of information. Subtracting the elapsed time pushes towards formulas that are quick to evaluate.

## What worked

- **Compression as a stand-in for interest.** PNG bytes per pixel of the bailout image rejects flat and noisy-but-boring
  pictures cheaply, at 64x64. This is the same idea as the video-size score in MatrixFractals.
- **Rank normalisation of exit values** (`ComplexMatrix`). Raw escape values have an enormous dynamic range; ranking makes
  every picture use the whole colormap.
- **Two-stage rendering.** Search at 64x64, redraw winners at up to 4000x4000 from the saved genome (`FractalUpscaler`).
  Winners are stored as genomes, not pixels, so any size can be regenerated.
- **An archive.** The 29,700 genomes were reusable as a pool and for comparison with later runs.
- **Genome diversity from function types.** Seven function families give the search far more shapes than polynomials alone.
- **Nested hyperparameter tuning.** Optuna with a mutation-probability list per gene type (30 values) and a vocabulary size
  was set up for parallel processes (`parallel.sh` starts five).

## What did not work

- **Most random genomes are useless.** About 1 in 20 random genomes passed validity in a quick check with the GUI
  (2 of 40). That is why the GA starts from many individuals and why `Miner.py` loops forever.
- **Rewriting collapses to a fixed point.** In a random sample of 1,500 saved runs, **1,472 (98%) end with two identical
  consecutive lines**, with a median final length of 12 characters. So after a few steps the rules stop changing the string and
  the "evolving formula" is in effect a single short formula. The long rewriting rules are mostly dead weight *(inferred)*; the
  search is closer to evolving a short formula plus constants than a grammar.
- **A stuck population.** Many `.txt` files repeat one line for all 100 iterations, and the code contains a stagnation detector
  that restarts the population *(inferred: the population converged often)*.
- **Elapsed time inside the fitness.** It makes the score depend on the machine load, so two runs of the same genome differ,
  and with five processes in parallel the penalty depends on the neighbours.
- **Autoencoder class scoring.** A VAE with a classifier was trained on fractal classes sorted into folders (names like `bacteroid`,
  `Hyperboid`, `equilibroid`) and used for a while to reward or penalise classes. It was dropped from the final fitness, and it
  needs a 150-300 MB model, a class folder and machine-specific paths.
- **Rule-structure scores.** Triangle growth of the line lengths, the Cython symmetry score, string compression ratio, HOG
  diversity, GLCM and wavelet statistics: all coded, none kept as the main objective (commented out in `evaluate`).
- **`eaSimple`.** Replaced by a hand-written loop that selects from offspring and parents together.
- **Reuse of past winners.** `generate_individual` has the "start from the best pool" branch disabled with `if False`.
- **`Miner.py`** instantiates `Fractal()` and `RewriterRandomGenerator.setrandomconstants()` with the old signatures, so it no longer runs.
- **A credential in the source.** The database password sat in `StudyManager.py`; it was removed before publication and the
  storage URL now comes from `GAOPTUNA_STORAGE` (default: local SQLite). If you reused that password anywhere, change it.

## Lessons

1. Evolve the *string and constants* together only if the rewriting does not collapse; check for fixed points and reward growth
   over iterations, or the grammar does no work.
2. Never put wall-clock time in a fitness unless runs are meant to be incomparable.
3. A cheap compression score at low resolution is enough to find candidates; save the genome so you can re-render later.
4. Keep the number of fitness terms small. Every experiment here ended up back at one image-complexity term.
5. Use rank normalisation for escape values.

## Running the original code

`src/GeneticAlgorithm.py` is the entry point (Optuna study named in its `__main__`). Run it from the repository root so that `src.X`
imports resolve. It needs DEAP, Optuna, jsonpickle, scikit-image, scikit-learn and the autoencoder weights (which are not in the repository).
`parallel.sh` launches five copies. These scripts contain absolute paths from the author's machine.

## Data formats

- `.json`: `jsonpickle.encode([start_string, rules, constants])`; tensors are embedded and were saved on CUDA.
- `.txt`: the rewritten lines, one per iteration.
- PNG: file names give the timestamp and resolution (`176452688870r4000x4000.png`).
