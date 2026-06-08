"""
====================================================================
AI-Driven Autonomous Threat Hunting in Wireless Networks
SVM-Based Intrusion Detection System — Simulation Code

Module: 25CSCN07I  |  Student: Abdelrahman Shehta  |  ID: 243700
====================================================================

Description:
    Simulates an SVM-based IDS across different 5G network sizes
    (10–50 nodes). Synthetic traffic is generated for normal, DoS,
    spoofing, and unauthorized-access scenarios. The SVM model is
    trained and evaluated per network configuration, with results
    reported across four metrics:
        - Detection Accuracy
        - False Positive Rate
        - Average Response Time
        - Computational Overhead
"""

import time
import warnings
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
NETWORK_SIZES   = [10, 20, 30, 40, 50]   # nodes
SIM_ROUNDS      = 50                      # rounds per config
SAMPLES_PER_NODE = 100                    # traffic samples per node
ATTACK_RATIO    = 0.35                    # 35% malicious traffic
TRAIN_RATIO     = 0.70                    # 70/30 split

# ─────────────────────────────────────────────────────────────────
# TRAFFIC GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_5g_traffic(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 5G-like network traffic features.

    Features (8 per sample):
        0: packet_size      — byte count of individual packets
        1: packet_rate      — packets per second
        2: inter_arrival    — ms between successive packets
        3: flow_duration    — connection flow duration (s)
        4: protocol_entropy — Shannon entropy of protocol mix
        5: port_diversity   — ratio of unique dest ports
        6: payload_ratio    — payload / header size ratio
        7: burst_flag       — binary flag for burst traffic
    """
    n_samples  = n_nodes * SAMPLES_PER_NODE
    n_attack   = int(n_samples * ATTACK_RATIO)
    n_normal   = n_samples - n_attack

    # Normal traffic — stationary Gaussian-ish distributions
    normal = np.column_stack([
        np.random.normal(512,  100, n_normal),      # packet_size
        np.random.normal(50,   10,  n_normal),      # packet_rate
        np.random.normal(20,   5,   n_normal),      # inter_arrival
        np.random.normal(30,   8,   n_normal),      # flow_duration
        np.random.normal(0.6,  0.1, n_normal),      # protocol_entropy
        np.random.normal(0.4,  0.1, n_normal),      # port_diversity
        np.random.normal(0.7,  0.1, n_normal),      # payload_ratio
        np.random.binomial(1, 0.05, n_normal),      # burst_flag
    ])

    # Attack traffic — shifted distributions to model threat behavior
    # Mix of DoS, Spoofing, Unauthorized Access characteristics
    attack_type = np.random.choice([0, 1, 2], n_attack,
                                   p=[0.45, 0.30, 0.25])  # DoS / Spoof / UA

    def dos_features(n):
        return np.column_stack([
            np.random.normal(64,   15,  n),         # small packets (flood)
            np.random.normal(5000, 800, n),         # very high packet rate
            np.random.normal(0.2,  0.05, n),        # tiny inter-arrival
            np.random.normal(2,    0.5, n),         # short flows
            np.random.normal(0.2,  0.05, n),        # low entropy
            np.random.normal(0.05, 0.02, n),        # single port
            np.random.normal(0.1,  0.05, n),        # almost no payload
            np.ones(n),                             # burst = always
        ])

    def spoof_features(n):
        return np.column_stack([
            np.random.normal(520,  120, n),         # similar to normal
            np.random.normal(55,   12,  n),
            np.random.normal(18,   6,   n),
            np.random.normal(25,   7,   n),
            np.random.normal(0.9,  0.05, n),        # high entropy (spoofed src)
            np.random.normal(0.8,  0.1,  n),        # many ports
            np.random.normal(0.65, 0.1,  n),
            np.random.binomial(1, 0.3, n),
        ])

    def ua_features(n):
        return np.column_stack([
            np.random.normal(256,  80,  n),         # smaller packets
            np.random.normal(20,   8,   n),         # moderate rate
            np.random.normal(40,   12,  n),         # longer inter-arrival
            np.random.normal(120,  30,  n),         # long flows (persistence)
            np.random.normal(0.5,  0.1, n),
            np.random.normal(0.6,  0.15, n),        # diverse ports (scanning)
            np.random.normal(0.4,  0.1,  n),
            np.random.binomial(1, 0.1, n),
        ])

    dos_idx   = np.where(attack_type == 0)[0]
    spoof_idx = np.where(attack_type == 1)[0]
    ua_idx    = np.where(attack_type == 2)[0]

    attack = np.zeros((n_attack, 8))
    if len(dos_idx):   attack[dos_idx]   = dos_features(len(dos_idx))
    if len(spoof_idx): attack[spoof_idx] = spoof_features(len(spoof_idx))
    if len(ua_idx):    attack[ua_idx]    = ua_features(len(ua_idx))

    X = np.vstack([normal, attack])
    y = np.array([0] * n_normal + [1] * n_attack)   # 0=normal, 1=attack

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────
# SINGLE SIMULATION ROUND
# ─────────────────────────────────────────────────────────────────
def run_simulation_round(n_nodes: int) -> dict:
    """Train and evaluate SVM on one round of synthetic traffic."""
    X, y = generate_5g_traffic(n_nodes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_RATIO, random_state=None, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Measure response time: only inference, not training
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_train, y_train)

    t0   = time.perf_counter()
    pred = model.predict(X_test)
    t1   = time.perf_counter()

    response_ms = (t1 - t0) * 1000  # ms for entire test batch

    acc = accuracy_score(y_test, pred) * 100
    cm  = confusion_matrix(y_test, pred)

    # FPR = FP / (FP + TN)  [normal samples misclassified as attack]
    tn, fp, fn, tp = cm.ravel()
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0

    return {"accuracy": acc, "fpr": fpr, "response_ms": response_ms}


# ─────────────────────────────────────────────────────────────────
# FULL SIMULATION ACROSS NETWORK SIZES
# ─────────────────────────────────────────────────────────────────
def run_full_simulation() -> dict:
    results = {}
    base_time = None  # for overhead normalization

    print("=" * 60)
    print("  AI-Driven Threat Hunting — SVM IDS Simulation")
    print(f"  Rounds per config: {SIM_ROUNDS}  |  Attack ratio: {ATTACK_RATIO:.0%}")
    print("=" * 60)

    for n_nodes in NETWORK_SIZES:
        print(f"\n[+] Network size: {n_nodes} nodes  ({n_nodes * SAMPLES_PER_NODE} traffic samples)")

        acc_list  = []
        fpr_list  = []
        time_list = []

        t_wall_start = time.perf_counter()
        for r in range(SIM_ROUNDS):
            res = run_simulation_round(n_nodes)
            acc_list.append(res["accuracy"])
            fpr_list.append(res["fpr"])
            time_list.append(res["response_ms"])

        t_wall_end = time.perf_counter()
        wall_time  = t_wall_end - t_wall_start  # total wall time for overhead proxy

        if base_time is None:
            base_time = wall_time

        mean_acc  = np.mean(acc_list)
        mean_fpr  = np.mean(fpr_list)
        mean_resp = np.mean(time_list)
        overhead  = wall_time / base_time

        results[n_nodes] = {
            "accuracy":    round(mean_acc, 1),
            "fpr":         round(mean_fpr, 1),
            "response_ms": round(mean_resp, 1),
            "overhead":    round(overhead, 2),
        }

        print(f"    Accuracy   : {mean_acc:.1f}%")
        print(f"    FPR        : {mean_fpr:.1f}%")
        print(f"    Resp. Time : {mean_resp:.1f} ms (avg over test batch)")
        print(f"    Overhead   : {overhead:.2f}x")

    return results


# ─────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────
def plot_results(results: dict, save_path: str = "simulation_results.png"):
    nodes    = list(results.keys())
    accuracy = [results[n]["accuracy"]    for n in nodes]
    fpr      = [results[n]["fpr"]         for n in nodes]
    resp     = [results[n]["response_ms"] for n in nodes]
    overhead = [results[n]["overhead"]    for n in nodes]

    NAVY  = "#0D1B2A"
    CYAN  = "#00C2CB"
    RED   = "#E84855"
    AMBER = "#F39C12"
    GRAY  = "#8BA7B8"

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(
        "SVM-Based IDS Performance — 5G Network Simulation",
        fontsize=16, fontweight="bold", color=NAVY, y=0.97
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.08, right=0.97, top=0.92, bottom=0.08)

    def styled_ax(ax, title, ylabel, color):
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=10)
        ax.set_xlabel("Network Size (nodes)", fontsize=10, color=GRAY)
        ax.set_ylabel(ylabel, fontsize=10, color=GRAY)
        ax.set_facecolor("#F8FBFF")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRAY)
        ax.tick_params(colors=GRAY)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_xticks(nodes)

    # 1 — Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    styled_ax(ax1, "Detection Accuracy vs. Network Size", "Accuracy (%)", CYAN)
    ax1.plot(nodes, accuracy, "o-", color=CYAN, linewidth=2.5, markersize=7, markerfacecolor=NAVY)
    ax1.set_ylim(85, 100)
    for x, y_val in zip(nodes, accuracy):
        ax1.annotate(f"{y_val}%", (x, y_val), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=NAVY, fontweight="bold")

    # 2 — FPR
    ax2 = fig.add_subplot(gs[0, 1])
    styled_ax(ax2, "False Positive Rate vs. Network Size", "FPR (%)", RED)
    bars = ax2.bar(nodes, fpr, color=RED, alpha=0.85, width=5, zorder=3)
    for bar, val in zip(bars, fpr):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{val}%", ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold")

    # 3 — Response Time
    ax3 = fig.add_subplot(gs[1, 0])
    styled_ax(ax3, "Avg. Response Time vs. Network Size", "Response Time (ms)", AMBER)
    ax3.plot(nodes, resp, "s--", color=AMBER, linewidth=2.5, markersize=7, markerfacecolor=NAVY)
    ax3.fill_between(nodes, resp, alpha=0.12, color=AMBER)
    for x, y_val in zip(nodes, resp):
        ax3.annotate(f"{y_val}ms", (x, y_val), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=NAVY)

    # 4 — Overhead
    ax4 = fig.add_subplot(gs[1, 1])
    styled_ax(ax4, "Computational Overhead vs. Network Size", "Overhead (normalized)", NAVY)
    ax4.plot(nodes, overhead, "D-", color=NAVY, linewidth=2.5, markersize=7, markerfacecolor=CYAN)
    ax4.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.2, label="Baseline (10 nodes)")
    ax4.legend(fontsize=9, framealpha=0.6)
    for x, y_val in zip(nodes, overhead):
        ax4.annotate(f"{y_val}×", (x, y_val), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=NAVY)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[✔] Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
def print_summary_table(results: dict):
    print("\n" + "=" * 65)
    print(f"  {'Nodes':>6}  {'Accuracy (%)':>14}  {'FPR (%)':>9}  "
          f"{'Resp.(ms)':>11}  {'Overhead':>10}")
    print("-" * 65)
    for n, r in results.items():
        print(f"  {n:>6}  {r['accuracy']:>14.1f}  {r['fpr']:>9.1f}  "
              f"{r['response_ms']:>11.1f}  {r['overhead']:>10.2f}")
    print("=" * 65)
    print("\n  Key observations:")
    nodes = list(results.keys())
    best  = results[nodes[0]]
    worst = results[nodes[-1]]
    print(f"  • Accuracy dropped  {best['accuracy']}% → {worst['accuracy']}%  "
          f"(-{best['accuracy'] - worst['accuracy']:.1f} pp)")
    print(f"  • FPR worsened      {best['fpr']}% → {worst['fpr']}%  "
          f"(+{worst['fpr'] - best['fpr']:.1f} pp)")
    print(f"  • Overhead grew     {best['overhead']}× → {worst['overhead']}×\n")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SVM IDS simulation")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick smoke-test (fewer rounds/samples)")
    args = parser.parse_args()

    if args.quick:
        SIM_ROUNDS = 3
        SAMPLES_PER_NODE = 20

    results = run_full_simulation()
    print_summary_table(results)
    plot_results(results, save_path="simulation_results.png")
