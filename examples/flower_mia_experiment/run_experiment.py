"""
Coalition Membership Inference Experiment
=========================================
Runs federated rounds manually (no Flower simulation) so we have full control
over which clients participate in each coalition.

For each (sigma, clip_norm, coalition_size) configuration:
  - Run many rounds with balanced coalitions (half include target, half exclude)
  - Aggregate client updates via FedAvg
  - Apply post-aggregation DP (clip + Gaussian noise)
  - Compute task AUC and accuracy on held-out validation set
  - Extract attacker features from the released model on a probe set
  - Train an attacker to predict coalition membership → attack AUC
  - Compare each DP run against the matching no-DP accuracy baseline
"""

import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from data import load_data
from models import LogisticRegression
from coalition_schedule import build_coalition_schedule
from dp_utils import compute_model_delta, clip_update, add_gaussian_noise, apply_noisy_delta
from eval_utils import evaluate_task_metrics
from attacker import extract_attack_features, compute_attack_auc

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters(model):
    return [val.cpu().detach().numpy().copy() for val in model.parameters()]


def set_parameters(model, parameters):
    params_dict = zip(model.parameters(), parameters)
    for p, new_val in params_dict:
        p.data = torch.tensor(new_val, dtype=p.dtype)


def client_train(model, trainloader, epochs=1, lr=0.01):
    """Train *model* on *trainloader* for a few local epochs and return updated params."""
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return get_parameters(model), len(trainloader.dataset)


def fedavg_aggregate(results):
    """Weighted average of client parameters (FedAvg)."""
    total_examples = sum(n for _, n in results)
    new_params = None
    for params, n in results:
        weight = n / total_examples
        if new_params is None:
            new_params = [p * weight for p in params]
        else:
            new_params = [np + p * weight for np, p in zip(new_params, params)]
    return new_params


def add_no_dp_accuracy_columns(df):
    baseline = (
        df[(df["sigma"] == 0.0) & (df["clip_norm"] == 0.0)][
            ["coalition_size", "task_accuracy"]
        ]
        .drop_duplicates(subset=["coalition_size"])
        .rename(columns={"task_accuracy": "no_dp_task_accuracy"})
    )
    if baseline.empty:
        raise ValueError(
            "Expected at least one baseline run with sigma=0.0 and clip_norm=0.0."
        )

    df = df.merge(baseline, on=["coalition_size"], how="left")
    df["no_dp_task_accuracy"] = df["no_dp_task_accuracy"].round(4)
    df["task_accuracy_delta_vs_no_dp"] = (
        df["task_accuracy"] - df["no_dp_task_accuracy"]
    ).round(4)

    return df[
        [
            "sigma",
            "clip_norm",
            "coalition_size",
            "no_dp_task_accuracy",
            "task_accuracy",
            "task_accuracy_delta_vs_no_dp",
            "task_auc",
            "attack_auc",
        ]
    ]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(sigma, clip_norm, coalition_size, target_id=0,
                   num_clients=10, num_rounds=200, local_epochs=1, lr=0.01,
                   seed=42):

    print(f"--- sigma={sigma}, clip={clip_norm}, coalition_size={coalition_size} ---")
    set_experiment_seed(seed)

    # 1. Data
    client_loaders, val_loader, probe_loader = load_data(num_clients)

    # 2. Coalition schedule (uses int IDs)
    client_ids = list(range(num_clients))
    schedule = build_coalition_schedule(client_ids, target_id, num_rounds, coalition_size)

    # 3. Global model
    global_model = LogisticRegression(input_dim=20)
    global_params = get_parameters(global_model)

    # 4. Run rounds
    attack_features_list = []
    attack_labels = []
    task_aucs = []
    task_accuracies = []

    for rnd, coalition in enumerate(schedule, 1):
        # ---- client training ----
        results = []
        for cid in coalition:
            local_model = LogisticRegression(input_dim=20)
            set_parameters(local_model, global_params)            # start from global
            updated, n_examples = client_train(local_model, client_loaders[cid],
                                               epochs=local_epochs, lr=lr)
            results.append((updated, n_examples))

        # ---- aggregation ----
        aggregated = fedavg_aggregate(results)

        # ---- DP: clip + noise the *update* ----
        if clip_norm <= 0:
            if sigma != 0:
                raise ValueError("clip_norm must be > 0 when sigma is non-zero.")
            released = aggregated
        else:
            delta = compute_model_delta(global_params, aggregated)
            clipped = clip_update(delta, clip_norm)
            noisy_delta = add_gaussian_noise(clipped, sigma, clip_norm)
            released = apply_noisy_delta(global_params, noisy_delta)

        # advance global state
        global_params = released

        # ---- metrics ----
        set_parameters(global_model, released)
        task_metrics = evaluate_task_metrics(global_model, val_loader)
        task_auc = task_metrics["auc"]
        task_accuracy = task_metrics["accuracy"]
        task_aucs.append(task_auc)
        task_accuracies.append(task_accuracy)

        feats = extract_attack_features(global_model, probe_loader)
        attack_features_list.append(feats)
        attack_labels.append(1 if target_id in coalition else 0)

        if rnd % 50 == 0 or rnd == 1:
            print(
                f"  round {rnd}/{num_rounds}  "
                f"task_auc={task_auc:.4f}  task_accuracy={task_accuracy:.4f}"
            )

    # 5. Attack AUC
    X_attack = np.array(attack_features_list)
    y_attack = np.array(attack_labels)
    attack_auc = compute_attack_auc(X_attack, y_attack)

    avg_task_auc = float(np.mean(task_aucs[-20:]))
    avg_task_accuracy = float(np.mean(task_accuracies[-20:]))

    print(
        f"  → task_auc={avg_task_auc:.4f}  "
        f"task_accuracy={avg_task_accuracy:.4f}  attack_auc={attack_auc:.4f}\n"
    )

    return {
        "sigma": sigma,
        "clip_norm": clip_norm,
        "coalition_size": coalition_size,
        "task_accuracy": round(avg_task_accuracy, 4),
        "task_auc": round(avg_task_auc, 4),
        "attack_auc": round(attack_auc, 4),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = []

    experiments = [
        {"sigma": 0.0,  "clip_norm": 0.0, "coalition_size": 10},
        {"sigma": 0.25, "clip_norm": 1.0, "coalition_size": 10},
        {"sigma": 0.5,  "clip_norm": 1.0, "coalition_size": 10},
        {"sigma": 1.0,  "clip_norm": 1.0, "coalition_size": 10},
        {"sigma": 2.0,  "clip_norm": 1.0, "coalition_size": 10},
    ]

    for exp in experiments:
        res = run_experiment(
            sigma=exp["sigma"],
            clip_norm=exp["clip_norm"],
            coalition_size=exp["coalition_size"],
            target_id=0,
            num_clients=20,
            num_rounds=200,
        )
        results.append(res)

    df = pd.DataFrame(results)
    df = add_no_dp_accuracy_columns(df)
    print("\n===== Final Results =====")
    print(df.to_markdown(index=False))
    df.to_csv("examples/flower_mia_experiment/mia_dp_results.csv", index=False)
    print("\nSaved to examples/flower_mia_experiment/mia_dp_results.csv")
