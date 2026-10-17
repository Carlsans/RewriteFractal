# Features that exist in the code but are not part of the final workflow

All are reachable from the GUI. "Evolve" holds options that change the GA; "Legacy" holds standalone tools.

| # | Feature | Where in the code | Why unused | In the GUI |
|---|---|---|---|---|
| 1 | Fitness: exit-value PNG size | `GeneticAlgorithm.evaluate` | Replaced by bailout PNG size | Evolve > Fitness (`exit_png`) |
| 2 | Fitness: Haar wavelet variance and entropy | `ClaudesMethod.wavelet_coefficient_stats` | Computed, then commented out of the sum | Evolve > Fitness (`wavelet`) |
| 3 | Fitness: gzip ratio of the rewritten lines | `RewriterEvaluator.evaluateRewriterByCompressionRatio` | Commented out | Evolve > Fitness (`string_compression`) |
| 4 | Fitness: triangle growth of line lengths | `RewriterEvaluator.simpletrianglescore` and relatives | Commented out | Evolve > Fitness (`triangle`) |
| 5 | Fitness: distance to the bundled Einstein image | `Sieve.comparetootherimagenote` | Experiment | Evolve > Fitness (`einstein_distance`) |
| 6 | Fitness: autoencoder class match | `Sieve.getimageautoencoderclasscorrespondance`, `src/autoencoder` | Needs a trained model and class folders; dropped | Evolve > Fitness (`autoencoder_class`), class names in the legacy box; reports why it is unavailable if the model is missing |
| 7 | Time penalty | `GeneticAlgorithm.evaluate` | Was in the last runs, makes results non-reproducible | Evolve > Legacy checkbox |
| 8 | Select from offspring only (`eaSimple`) | commented block at the end of `GeneticAlgorithm.main` | Replaced by selecting from offspring + parents | Evolve > Legacy checkbox |
| 9 | Start from past winners | `PoolManager.getBestPastIndividuals`, `if False` in `generate_individual` | Disabled | Evolve > Legacy "Seed from folder" |
| 10 | Stagnation restart | `GeneticAlgorithm.detect_fitness_stagnation` | Kept, but the original restarted the process | Evolve checkbox |
| 11 | Random miner | `Miner.py` | Broken (old signatures) | Legacy > Random miner |
| 12 | Taylor-series variations | `GeneticAlgorithm.draw_individual_with_variations` | Experiment, unused call | Legacy > Taylor-series variations |
| 13 | Optuna hyperparameter study | `StudyManager.py` | Needed a MySQL server | Legacy > Hyperparameter study (in memory, or `GAOPTUNA_STORAGE`) |
| 14 | Batch upscaling | `FractalUpscaler.py` | Fixed folder, script-style | Legacy > Batch upscale |
| 15 | Cython symmetry score | `RewriterEngineCython.mesureSymetry` | Commented out; needs a compiled extension | Not integrated (needs `rwsetup.py` build; no known good use) |
| 16 | Website ownership timestamps | `Website/WebsiteScripts.py` | Separate purpose (proof of authorship) | Not integrated |
| 17 | HOG diversity, GLCM, LBP, mutual information, fractal dimension | `RewriterEvaluator`, `ClaudesMethod` | Written and compared, never selected | Not integrated: wavelet stands in for the group; the rest need heavy extra dependencies and were never found useful |
