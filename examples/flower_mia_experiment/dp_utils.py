import numpy as np

def compute_model_delta(global_before, global_after):
    return [b - a for a, b in zip(global_before, global_after)]

def clip_update(delta, clip_norm):
    flat_norm = np.sqrt(sum(np.sum(d * d) for d in delta))
    scale = min(1.0, clip_norm / (flat_norm + 1e-12))
    return [d * scale for d in delta]

def add_gaussian_noise(delta, sigma, clip_norm):
    std = sigma * clip_norm
    return [d + np.random.normal(0, std, size=d.shape) for d in delta]

def apply_noisy_delta(global_before, noisy_delta):
    return [w + d for w, d in zip(global_before, noisy_delta)]
