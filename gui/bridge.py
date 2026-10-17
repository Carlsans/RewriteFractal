"""Glue between the PySide6 GUI and the RewriteFractal engine.

The engine lives in ``src/`` and imports itself as ``src.X``. ``src.Fractal`` imports ``src.Sieve``, which loads a
trained autoencoder at construction. The GUI does not need that model for the metrics the GA last used, so a light
``Sieve`` with the same image-size metric is registered instead. The autoencoder metrics are still available
(``autoencoder_class``) when a model and class folder exist.

A genome is the engine's own individual: ``[start_string, rules (OrderedDict), constants (OrderedDict)]``,
saved with jsonpickle exactly like ``GeneticAlgorithm.draw_individual`` so existing ``.json`` files open.
"""
import copy
import gzip
import io
import math
import os
import random
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
OPERATORS = "+-*/"
DEFAULT_OPTIONAL_VOCAB = "ABDEFGHIJKLMNOPQRSTUVWRXY"
DEFAULT_FN_VOCAB = "abdefghijklmnopqrstuvwrxy"
COLORMAPS = ['gist_earth', 'terrain', 'gist_stern', 'nipy_spectral', 'cubehelix', 'gnuplot',
             'gnuplot2', 'CMRmap', 'ocean', 'turbo', 'twilight', 'binary']
METRICS = OrderedDict([
    ("bailout_png", "Bailout PNG size per pixel (active in the last GA runs)"),
    ("exit_png", "Exit-value PNG size per pixel (legacy)"),
    ("wavelet", "Haar wavelet variance + entropy of exit values (legacy)"),
    ("string_compression", "gzip ratio of the rewritten lines (legacy)"),
    ("triangle", "Triangle growth of rewritten lines (legacy)"),
    ("einstein_distance", "Closeness to the bundled Einstein image (legacy)"),
    ("autoencoder_class", "Autoencoder class match (legacy, needs a trained model and class folders)"),
])


class _Sieve:
    """Stand-in for src.Sieve.Sieve: same image-information metric, no autoencoder."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def getimageinformationsnote(self, tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.cpu()
        return self.save_image_to_memory_and_measure(np.uint8(255 * tensor))

    def save_image_to_memory_and_measure(self, array_data):
        if array_data.sum() == 0:
            return None
        pil_image = Image.fromarray(array_data)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return buffer.tell() / (pil_image.size[0] * pil_image.size[1])

    def isbailouttensorvalid(self, tensor):
        return not torch.min(tensor) == torch.max(tensor)


def install_shims():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if "src.Sieve" not in sys.modules:
        mod = types.ModuleType("src.Sieve")
        mod.Sieve = _Sieve
        sys.modules["src.Sieve"] = mod
    try:
        import Crypto.Random.random  # noqa: F401
    except ImportError:
        # RewriterRandomGenerator uses pycryptodome's rng; fall back to the standard one.
        crypto = types.ModuleType("Crypto")
        rnd_pkg = types.ModuleType("Crypto.Random")
        rnd = types.ModuleType("Crypto.Random.random")
        rnd.randrange, rnd.choice, rnd.randint = random.randrange, random.choice, random.randint
        crypto.Random, rnd_pkg.random = rnd_pkg, rnd
        sys.modules.update({"Crypto": crypto, "Crypto.Random": rnd_pkg, "Crypto.Random.random": rnd})


install_shims()

import jsonpickle  # noqa: E402

from src.Fractal import Fractal  # noqa: E402
from src.GAHyperParameters import GAHyperParameters  # noqa: E402
from src.RewriterEngine import RewriterEngine  # noqa: E402
from src.RewriterRandomGenerator import RewriterRandomGenerator  # noqa: E402
from src.functions.Trigonometric import Trigonometric  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIEVE = _Sieve()


# ---------------------------------------------------------------- genomes
def load_genome(path):
    """Open a genome .json (jsonpickle). Tensors stored on CUDA need CUDA to load."""
    with open(path, "r") as handle:
        individual = jsonpickle.decode(handle.read())
    for key, value in list(individual[2].items()):
        if key.islower() and len(key) == 1 and not isinstance(value, dict) and torch.is_tensor(value):
            individual[2][key] = {'fntype': 'Taylorian', 'value': value.to(DEVICE)}  # old files
    return [individual[0], OrderedDict(individual[1]), OrderedDict(individual[2])]


def save_genome(genome, path):
    with open(path, "w") as handle:
        handle.write(jsonpickle.encode([genome[0], genome[1], genome[2]]))


def make_hp(**overrides):
    hp = GAHyperParameters()
    hp.polynomialcoeffmultiplier, hp.polynomialexposant = 1.0, 5
    hp.rationalnumeratorcoeffmultiplier, hp.rationalnumeratorexposant = 1.0, 5
    hp.rationaldenominatorcoeffmultiplier, hp.rationaldenominatorexposant = 1.0, 5
    for key, value in overrides.items():
        setattr(hp, key, value)
    return hp


def vocabulary_for(optional=4, functions=3):
    return "CZ" + DEFAULT_OPTIONAL_VOCAB[:optional] + DEFAULT_FN_VOCAB[:functions]


class Generator:
    """Wraps the engine's RewriterRandomGenerator for a vocabulary."""

    def __init__(self, vocabulary, hp=None):
        self.vocabulary = vocabulary
        self.hp = hp or make_hp()
        self.rng = RewriterRandomGenerator(vocabulary, OPERATORS)

    def new_genome(self):
        return [self.rng.randomstartstring(), self.rng.randomrules(), self.rng.setrandomconstants(self.hp)]


def describe_constant(value):
    if isinstance(value, dict):
        inner = value.get('value')
        if isinstance(inner, dict):
            parts = []
            for k, v in inner.items():
                parts.append(f"{k}={tuple(v.shape) if torch.is_tensor(v) and v.numel() > 1 else (v.item() if torch.is_tensor(v) else v)}")
            return f"{value['fntype']}({', '.join(parts)})"
        if torch.is_tensor(inner):
            return f"{value['fntype']}({inner.numel()} terms)"
        return f"{value['fntype']}({inner})"
    if isinstance(value, (list, tuple)):
        return ", ".join(f"{v:.4g}" for v in value)
    if isinstance(value, complex):
        return f"{value.real:+.4f}{value.imag:+.4f}j"
    return str(value)


# ---------------------------------------------------------------- rendering
class Rendered:
    def __init__(self, rewriter, fractal):
        self.rewriter, self.fractal = rewriter, fractal


def build(genome, resolution=(512, 512), maxlinelength=20000, use_exit_values=True):
    """Rewrite, check, run and return Rendered, or raise ValueError with the reason."""
    rewriter = RewriterEngine(genome[0], genome[1], genome[2], None, maxiter=100, maxlinelenght=maxlinelength)
    if not rewriter.rewrite():
        raise ValueError(f"rewritten line exceeded {maxlinelength} characters")
    if not rewriter.sanitycheck():
        raise ValueError("sanity check failed (too many functions or Z/C on a line)")
    fractal = Fractal(SIEVE, center=rewriter.getFractalCenter(), range=rewriter.getFractalRange(),
                      resolution=resolution, useexitbailoutvalues=use_exit_values)
    rewriter.fractal = fractal
    rewriter.calculateConstants()
    rewriter.runAllLines()
    return Rendered(rewriter, fractal)


def to_rgb(rendered, colormap="gist_earth", source="exit"):
    from matplotlib import colormaps
    fractal = rendered.fractal
    if source == "exit":
        field = fractal.getNormalizedExitValues()
    else:
        field = fractal.bailout / torch.max(fractal.bailout).clamp(min=1)
    field = torch.nan_to_num(field.float()).cpu().numpy()
    rgba = colormaps.get_cmap(colormap)(field)
    return np.ascontiguousarray((rgba[..., :3] * 255).astype(np.uint8))


def render_genome(genome, resolution=(512, 512), colormap="gist_earth", source="exit", maxlinelength=20000):
    rendered = build(genome, resolution, maxlinelength)
    return to_rgb(rendered, colormap, source), rendered


# ---------------------------------------------------------------- metrics
_EINSTEIN = None


def _einstein(shape):
    global _EINSTEIN
    if _EINSTEIN is None:
        img = Image.open(SRC / "baceinsteinfullpng.png").convert("L")
        _EINSTEIN = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    resized = torch.nn.functional.interpolate(_EINSTEIN[None, None], size=tuple(shape), mode="bilinear",
                                              align_corners=False)
    return resized[0, 0]


def _shannon(t, bins=256):
    hist = torch.histc(t.float().flatten(), bins=bins, min=float(t.min()), max=float(t.max()) + 1e-9)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def _haar_wavelet(tensor):
    import torch.nn.functional as F
    x = tensor.float()[None, None]
    kernels = {
        'll': [[.5, .5], [.5, .5]], 'lh': [[.5, -.5], [.5, -.5]],
        'hl': [[.5, .5], [-.5, -.5]], 'hh': [[.5, -.5], [-.5, .5]],
    }
    out = {k: F.conv2d(x, torch.tensor(v, device=x.device)[None, None], stride=2) for k, v in kernels.items()}
    variance = sum(float(torch.var(v)) for v in out.values())
    entropy = sum(_shannon(out[k].squeeze()) for k in ('lh', 'hl', 'hh'))
    return variance, entropy


_AE = None


def _autoencoder():
    global _AE
    if _AE is None:
        from src.autoencoder.Inference import Inference
        _AE = Inference()
    return _AE


def metric_names_available():
    return list(METRICS.keys())


def score_genome(genome, metric="bailout_png", resolution=(64, 64), time_penalty=False, rendered=None,
                 class_targets=None):
    """Higher is better. Returns None when the genome is invalid or degenerate (the GA's penalty cases)."""
    started = time.time()
    try:
        rendered = rendered or build(genome, resolution, maxlinelength=200)
    except ValueError:
        return None
    fractal, rewriter = rendered.fractal, rendered.rewriter
    if fractal.zeroBailouttest():
        return None
    if metric == "bailout_png":
        value = SIEVE.getimageinformationsnote(fractal.bailout)
    elif metric == "exit_png":
        value = SIEVE.getimageinformationsnote(fractal.getNormalizedExitValues())
    elif metric == "wavelet":
        field = torch.nan_to_num(fractal.getNormalizedExitValues().float())
        variance, entropy = _haar_wavelet(field)
        value = variance + entropy
    elif metric == "string_compression":
        data = rewriter.getLinesAsString().encode("utf-8")
        value = len(gzip.compress(data)) / len(data) if data else None
    elif metric == "triangle":
        lines, depth, total, last = rewriter.stringlist, min(100, len(rewriter.stringlist) - 1), 0.0, None
        last = len(lines[0])
        for i in range(1, depth + 1):
            total += abs(len(lines[i]) - last - 2)
            last = len(lines[i])
        value = 1.0 - min(1.0, total / max(depth, 1) / 10.0)
    elif metric == "einstein_distance":
        field = torch.nan_to_num(fractal.getNormalizedExitValues().float()).cpu()
        mx, mn = field.max(), field.min()
        if mx == mn:
            return None
        field = (field - mn) / (mx - mn)
        value = -float((_einstein(field.shape) - field).flatten().pow(2).sum().sqrt() / field.shape[0])
    elif metric == "autoencoder_class":
        field = torch.nan_to_num(fractal.getNormalizedExitValues().float()).to(DEVICE)
        mx, mn = field.max(), field.min()
        if mx == mn:
            return None
        field = ((field - mn) / (mx - mn)).reshape(1, 1, *field.shape)
        if tuple(field.shape[-2:]) != (512, 512):
            field = torch.nn.functional.interpolate(field, size=(512, 512), mode="bilinear")
        positive, negative = (class_targets or ([], []))
        ae = _autoencoder()
        pos, neg = ae.getClassProbability(field, positive, negative, 15)
        value = float(sum(torch.relu(p).item() for p in pos) - sum(torch.relu(n).item() for n in neg))
    else:
        raise ValueError(metric)
    if value is None:
        return None
    if time_penalty:
        value -= (time.time() - started)
    return float(value)


# ---------------------------------------------------------------- GA operators (ported from GeneticAlgorithm.py)
def random_special(endian=None):
    if endian is None:
        endian = random.random() < 0.5
    if endian:
        return random.random() * 2 - 1
    return 1 / (1 - random.random() * 2 + 1e-12)


def cross_strings(s1, s2):
    if len(s1) > 1:
        point = random.randint(1, len(s1) - 1)
        return s1[:point] + s2[point:], s2[:point] + s1[point:]
    c = [random.randint(0, 1) for _ in range(6)]
    n1, n2 = s1 * c[0] + s2 * c[1], s2 * c[2] + s1 * c[3]
    if n1 == "":
        n1 = s1 * c[4] + s2 * abs(c[4] - 1)
    if n2 == "":
        n2 = s1 * c[5] + s2 * abs(c[5] - 1)
    return n1, n2


def crossover(ind1, ind2):
    """Start strings cut at a point; rule sets are paired and each pair crossed, extra rules
    are dealt out at random; constants are exchanged key by key."""
    a, b = copy.deepcopy(ind1), copy.deepcopy(ind2)
    if len(ind1[0]) > 1:
        point = random.randint(1, len(ind1[0]) - 1)
        a[0], b[0] = ind1[0][:point] + ind2[0][point:], ind2[0][:point] + ind1[0][point:]
    bigger, smaller = (a[1], b[1]) if len(a[1]) > len(b[1]) else (b[1], a[1])
    spare = OrderedDict()
    for _ in range(len(bigger) - len(smaller)):
        key = random.choice(list(bigger.keys()))
        spare[key] = bigger.pop(key)
    c1, c2 = OrderedDict(), OrderedDict()
    for (k1, v1), (k2, v2) in zip(smaller.items(), list(bigger.items())):
        nk, nv = cross_strings(k1, k2), cross_strings(v1, v2)
        c1[nk[0]], c2[nk[1]] = nv[0], nv[1]
    for key, value in spare.items():
        (c1 if random.randint(0, 1) else c2)[key] = value
    a[1], b[1] = c1, c2
    ca, cb = OrderedDict(), OrderedDict()
    for key in set(ind1[2]) | set(ind2[2]):
        pick_first = random.random() < 0.5
        src_a = ind1[2] if (key in ind1[2] and (pick_first or key not in ind2[2])) else ind2[2]
        src_b = ind2[2] if (key in ind2[2] and (pick_first or key not in ind1[2])) else ind1[2]
        ca[key], cb[key] = copy.deepcopy(src_a[key]), copy.deepcopy(src_b[key])
    a[2], b[2] = ca, cb
    return a, b


def _scale_tensor(t, endian, multiplier=None):
    multiplier = multiplier if multiplier is not None else random.choice([1 + 0j, 0 + 1j, 1 + 1j])
    return t * (random_special(endian) * multiplier)


def mutate(ind, p, vocabulary):
    """Gene-wise mutation of start string, rules, and every constant (probability p per gene)."""
    pool = vocabulary + OPERATORS
    chars = list(ind[0])
    for i in range(len(chars)):
        if random.random() < p:
            chars[i] = random.choice(pool)
    ind[0] = "".join(chars)
    new_rules = OrderedDict()
    for key, value in ind[1].items():
        k = "".join(random.choice(pool) if random.random() < p else c for c in key)
        v = "".join(random.choice(pool) if random.random() < p else c for c in value)
        new_rules[k] = v
    ind[1] = new_rules
    consts = ind[2]
    for key, value in consts.items():
        if key == "center":
            for idx in (0, 1):
                for endian in (False, True):
                    if random.random() < p:
                        value[idx] = value[idx] * random_special(endian)
            continue
        if key == "range":
            for endian in (False, True):
                if random.random() < p:
                    consts[key] = consts[key] * random_special(endian)
            continue
        if key.isupper():
            for part in ("real", "imag"):
                for endian in (False, True):
                    if random.random() < p:
                        c = consts[key]
                        f = random_special(endian)
                        consts[key] = complex(c.real * f, c.imag) if part == "real" else complex(c.real, c.imag * f)
            continue
        kind, inner = value['fntype'], value['value']
        if kind == 'Taylorian':
            for endian in (False, True):
                if random.random() < p:
                    idx = random.randint(0, inner.shape[0] - 1)
                    inner[idx] = _scale_tensor(inner[idx], endian)
        elif kind == 'Polynomial':
            for field in ('coefficients', 'exposants'):
                if random.random() < p:
                    idx = random.randint(0, inner[field].shape[0] - 1)
                    inner[field][idx] = _scale_tensor(inner[field][idx], random.random() < p)
        elif kind == 'Rational':
            for field in ('numeratorcoefficients', 'numeratorexposants', 'denominatorcoefficients', 'denominatorexposants'):
                if random.random() < p:
                    idx = random.randint(0, inner[field].shape[0] - 1)
                    inner[field][idx] = _scale_tensor(inner[field][idx], random.random() < p)
        elif kind in ('Power', 'Exponential', 'Logarithmic'):
            field = 'exponent' if kind == 'Power' else 'base'
            for endian in (False, True):
                if random.random() < p:
                    new = _scale_tensor(inner[field], endian)
                    inner[field] = abs(new) if kind == 'Logarithmic' else new
        elif kind == 'Trigonometric':
            if random.random() < p:
                inner['trigooperation'] = random.choice(Trigonometric().allowedoperations)
    return ind


class GAParams:
    def __init__(self):
        self.pop_size, self.crossing, self.mutation, self.tournament = 30, 0.7, 0.05, 3
        self.generations, self.resolution, self.metric = 20, 64, "bailout_png"
        self.optional_vocab, self.function_vocab = 4, 3
        self.time_penalty = False
        self.select_from_parents = True        # active code: select over offspring + population
        self.restart_on_stagnation = True      # detect_fitness_stagnation
        self.stagnation_generations = 15
        self.seed_folder = None                # legacy PoolManager seeding
        self.seed_mutations = 5
        self.class_targets = ([], [])


def tournament(population, scores, k, size):
    chosen = []
    for _ in range(k):
        pool = random.sample(range(len(population)), min(size, len(population)))
        chosen.append(max(pool, key=lambda i: scores[i]))
    return [copy.deepcopy(population[i]) for i in chosen]


def load_seed_pool(folder, limit=200):
    """Legacy PoolManager.getBestPastIndividuals: pick genomes from a folder of saved ones."""
    genomes = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.name.endswith(".json"):
                try:
                    genomes.append(load_genome(entry.path))
                except Exception:
                    continue
                if len(genomes) >= limit:
                    break
    return genomes


def run_ga(params, job=None, log=print, on_best=None):
    """Generational GA. Returns (best genome, best score, history).

    select_from_parents=True is the loop of GeneticAlgorithm.main (varAnd, then tournament over
    offspring + population); False is DEAP's eaSimple that the file kept in a comment.
    """
    generator = Generator(vocabulary_for(params.optional_vocab, params.function_vocab))
    vocabulary = generator.vocabulary
    res = (params.resolution, params.resolution)
    seeds = load_seed_pool(params.seed_folder) if params.seed_folder else []

    def new_individual():
        if seeds:
            ind = copy.deepcopy(random.choice(seeds))
            for _ in range(params.seed_mutations):
                mutate(ind, 0.01, vocabulary)
            return ind
        return generator.new_genome()

    def evaluate(ind):
        score = score_genome(ind, params.metric, res, params.time_penalty, class_targets=params.class_targets)
        return -1e9 if score is None else score

    population = [new_individual() for _ in range(params.pop_size)]
    scores = [evaluate(i) for i in population]
    best_score, best, history, stale = -1e18, None, [], 0
    for gen in range(params.generations):
        if job is not None and job.should_stop():
            break
        offspring = tournament(population, scores, params.pop_size, params.tournament)
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < params.crossing:
                offspring[i], offspring[i + 1] = crossover(offspring[i], offspring[i + 1])
        offspring = [mutate(o, params.mutation, vocabulary) if random.random() < params.mutation * 10 else o
                     for o in offspring]
        off_scores = [evaluate(o) for o in offspring]
        if params.select_from_parents:
            merged, merged_scores = offspring + population, off_scores + scores
        else:
            merged, merged_scores = offspring, off_scores
        order = sorted(range(len(merged)), key=lambda i: merged_scores[i], reverse=True)[:params.pop_size]
        population = [merged[i] for i in order]
        scores = [merged_scores[i] for i in order]
        history.append(scores[0])
        if scores[0] > best_score + 1e-9:
            best_score, best, stale = scores[0], copy.deepcopy(population[0]), 0
            if on_best:
                on_best(best, best_score)
        else:
            stale += 1
        valid = [x for x in scores if x > -1e8]
        log(f"gen {gen}: best {scores[0]:.4f}  valid {len(valid)}/{len(scores)}  global best {best_score:.4f}")
        if params.restart_on_stagnation and stale >= params.stagnation_generations:
            log("stagnation: new random population (the original returned and restarted the process)")
            population = [new_individual() for _ in range(params.pop_size)]
            scores = [evaluate(i) for i in population]
            stale = 0
    return best, best_score, history


def mine(metric_threshold=(0.1, 0.2), job=None, log=print, on_found=None, vocabulary="ABCZ", resolution=128):
    """Legacy Miner.mineFractals: random genomes, keep those whose note passes the thresholds."""
    generator = Generator(vocabulary)
    while job is None or not job.should_stop():
        genome = generator.new_genome()
        note = score_genome(genome, "bailout_png", (resolution, resolution))
        if note is None:
            continue
        log(f"note {note:.3f}")
        if note > metric_threshold[0] and on_found:
            on_found(genome, note, note > metric_threshold[1])


def taylor_variations(genome, key, resolution=(128, 128), colormap="gist_earth",
                      offsets=(-2j, -1j, -0.5j, -0.1j, 0.1j, 0.5j, 1j, 2j), max_terms=3):
    """Legacy draw_individual_with_variations: offset one term of a Taylorian constant at a time."""
    if key not in genome[2] or genome[2][key].get('fntype') != 'Taylorian':
        raise ValueError(f"constant {key!r} is not a Taylorian series")
    base = genome[2][key]['value']
    tiles = []
    for term in range(min(max_terms, base.shape[0])):
        for off in offsets:
            variant = copy.deepcopy(genome)
            variant[2][key]['value'][term] = base[term] + off
            try:
                rgb, _ = render_genome(variant, resolution, colormap)
            except Exception:
                rgb = np.zeros((*resolution, 3), np.uint8)
            tiles.append(rgb)
    return tiles, len(offsets)


def run_optuna_study(trials, generations, log=print, job=None, metric="bailout_png"):
    """Legacy StudyManager: tune GA hyperparameters with Optuna. Uses an in-memory (or $GAOPTUNA_STORAGE) study."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = os.environ.get("GAOPTUNA_STORAGE")
    study = optuna.create_study(direction="maximize", storage=storage, load_if_exists=bool(storage),
                                study_name="rewritefractal-gui" if storage else None)

    def objective(trial):
        if job is not None and job.should_stop():
            study.stop()
        p = GAParams()
        p.metric, p.generations, p.resolution = metric, generations, 48
        p.pop_size = trial.suggest_int("pop_size", 10, 60)
        p.mutation = trial.suggest_float("mutationpb", 0.001, 0.3)
        p.crossing = trial.suggest_float("crossingpb", 0.1, 1.0)
        p.tournament = trial.suggest_int("tournsize", 1, 10)
        p.optional_vocab = trial.suggest_int("optionalvocabulary", 1, 12)
        p.function_vocab = trial.suggest_int("taylorian_range", 1, 8)
        _, score, _ = run_ga(p, job=job, log=lambda s: None)
        log(f"trial {trial.number}: {score:.4f}  {trial.params}")
        return score

    study.optimize(objective, n_trials=trials)
    return study.best_params, study.best_value


def upscale_folder(folder, out_folder, size=(2000, 2000), colormap="gist_earth", job=None, log=print):
    """Legacy FractalUpscaler: re-render every genome in a folder at a large size."""
    from matplotlib import pyplot as plt
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for entry in sorted(Path(folder).glob("*.json")):
        if job is not None and job.should_stop():
            break
        try:
            rgb, _ = render_genome(load_genome(entry), size, colormap)
        except Exception as exc:
            log(f"{entry.name}: {exc}")
            continue
        plt.imsave(Path(out_folder) / f"{entry.stem}_{size[0]}x{size[1]}.png", rgb)
        log(f"wrote {entry.stem}")
