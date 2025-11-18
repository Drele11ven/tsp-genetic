# streamlit_app.py
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

# Ensure these modules exist in your project (from earlier code)
from ga import init_population, evolve, fitness_population
from utils import load_cities_from_csv, compute_distance_matrix, route_length

# -------------------------
# Configuration / Helpers
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def try_write_plotly_image(fig: go.Figure, path_png: str) -> bool:
    """
    Try to write a Plotly figure to a PNG. Return True if successful.
    Requires 'kaleido' to be installed. If it fails, write an HTML fallback.
    """
    try:
        # prefer PNG via kaleido
        pio.write_image(fig, path_png, format="png", engine="kaleido")
        return True
    except Exception as ex:
        # fallback: write interactive HTML
        try:
            fallback = path_png.replace(".png", ".html")
            pio.write_html(fig, fallback, include_plotlyjs='cdn')
            return False
        except Exception:
            return False


def save_run_results(
    run_timestamp: str,
    params: Dict[str, Any],
    best_route_df: pd.DataFrame,
    history_best: List[float],
    history_avg: List[float],
    fig_route: go.Figure
) -> str:
    """
    Save all outputs of a run into results/<timestamp>/ and return that path.
    Files saved:
      - config.json
      - best_route.csv
      - history.csv (generation,best,avg)
      - fitness_curve.png
      - route_plot.png (or route_plot.html if PNG fails)
    """
    out_dir = os.path.join(RESULTS_DIR, run_timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # 1. save params
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # 2. save best route CSV
    best_route_df.to_csv(os.path.join(out_dir, "best_route.csv"), index=False)

    # 3. save history
    df_hist = pd.DataFrame({
        "generation": list(range(1, len(history_best) + 1)),
        "best": history_best,
        "avg": history_avg
    })
    df_hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # 4. save fitness chart (matplotlib)
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(df_hist["generation"], df_hist["best"], label="best")
        plt.plot(df_hist["generation"], df_hist["avg"], label="avg")
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.title("Fitness Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fitness_curve.png"))
        plt.close()
    except Exception:
        # don't fail run if matplotlib plotting does not work
        try:
            plt.close()
        except Exception:
            pass

    # 5. save route plot (Plotly)
    route_png = os.path.join(out_dir, "route_plot.png")
    ok = try_write_plotly_image(fig_route, route_png)
    if not ok:
        # we still have an HTML fallback created in try_write_plotly_image
        pass

    return out_dir


def generate_professional_summary(output_dir: str = SUMMARY_DIR) -> Dict[str, Any]:
    """
    Aggregate all runs under results/ (except 'summary' dir) and produce:
      - results/summary/fitness_all_runs.png
      - results/summary/best_distance_boxplot.png
      - results/summary/runs_summary.csv
      - results/summary/best_run_info.json
    Returns a dict with summary statistics.
    """
    run_dirs = [
        os.path.join(RESULTS_DIR, d) for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d != "summary"
    ]
    runs_meta = []
    # Collect histories and final bests
    all_histories = []
    labels = []
    final_bests = []

    for rd in sorted(run_dirs):
        cfg_path = os.path.join(rd, "config.json")
        hist_path = os.path.join(rd, "history.csv")
        if not os.path.exists(cfg_path) or not os.path.exists(hist_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            hist_df = pd.read_csv(hist_path)
            best_final = float(hist_df["best"].iloc[-1])
            runs_meta.append({
                "run_folder": os.path.basename(rd),
                "config": cfg,
                "final_best": best_final,
                "generations": int(hist_df["generation"].iloc[-1])
            })
            all_histories.append(hist_df["best"].values)
            labels.append(os.path.basename(rd))
            final_bests.append(best_final)
        except Exception:
            continue

    # Create summary DataFrame
    summary_rows = []
    for meta in runs_meta:
        cfg_flat = meta["config"].copy()
        # flatten some config keys into columns
        summary_rows.append({
            "run_folder": meta["run_folder"],
            "final_best": meta["final_best"],
            "generations": meta["generations"],
            **{f"cfg_{k}": v for k, v in cfg_flat.items()}
        })
    summary_df = pd.DataFrame(summary_rows)
    runs_summary_csv = os.path.join(output_dir, "runs_summary.csv")
    summary_df.to_csv(runs_summary_csv, index=False)

    # Plot aggregated fitness curves (matplotlib)
    try:
        plt.figure(figsize=(10, 6))
        for idx, hist in enumerate(all_histories):
            plt.plot(range(1, len(hist) + 1), hist, label=labels[idx])
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.title("All Runs: Best Distance Over Generations")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        agg_png = os.path.join(output_dir, "fitness_all_runs.png")
        plt.savefig(agg_png)
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    # Boxplot of final best distances
    try:
        plt.figure(figsize=(6, 4))
        plt.boxplot(final_bests)
        plt.ylabel("Final Best Distance")
        plt.title("Distribution of Final Best Distances Across Runs")
        plt.tight_layout()
        box_png = os.path.join(output_dir, "best_distance_boxplot.png")
        plt.savefig(box_png)
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    # determine best run
    best_run_info = None
    if len(runs_meta) > 0:
        best_meta = min(runs_meta, key=lambda m: m["final_best"])
        best_run_info = {
            "best_run_folder": best_meta["run_folder"],
            "final_best": best_meta["final_best"],
            "config": best_meta["config"]
        }
        with open(os.path.join(output_dir, "best_run_info.json"), "w", encoding="utf-8") as f:
            json.dump(best_run_info, f, indent=2)

    return {
        "runs_count": len(runs_meta),
        "summary_csv": runs_summary_csv,
        "fitness_all_runs_png": os.path.join(output_dir, "fitness_all_runs.png"),
        "boxplot_png": os.path.join(output_dir, "best_distance_boxplot.png"),
        "best_run_info": best_run_info
    }


# -------------------------
# Streamlit UI / Main
# -------------------------
st.set_page_config(layout="wide")

st.title("TSP Solver — Genetic Algorithm (Professional Results Logging)")

st.markdown(
    "Interactive TSP solver using a Genetic Algorithm. "
    "Each run is logged under the `results/` folder (a timestamped subfolder). "
    "A professional summary is generated in `results/summary/` for cross-run analysis."
)

# Sidebar parameters
st.sidebar.header("Algorithm Parameters")
pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=5000, value=200, step=10)
generations = st.sidebar.number_input("Generations", min_value=1, max_value=10000, value=500, step=10)
elite_size = st.sidebar.number_input("Elite Size (kept)", min_value=0, max_value=200, value=5, step=1)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.02, step=0.001)
tournament_k = st.sidebar.number_input("Tournament k", min_value=2, max_value=20, value=3, step=1)
random_seed = st.sidebar.number_input("Random Seed (0 = none)", value=0)

st.sidebar.markdown("---")
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload cities CSV (id,name,x,y)", type=["csv"])
use_sample = st.sidebar.checkbox("Use Sample Dataset (data/cities_sample.csv)", value=True)
run_btn = st.sidebar.button("Run Algorithm")

# Load dataset
if uploaded:
    df = load_cities_from_csv(uploaded)
else:
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "cities_sample.csv")
    df = load_cities_from_csv(os.path.abspath(sample_path))

coords = df[['x', 'y']].to_numpy()
n_cities = coords.shape[0]
dist_matrix = compute_distance_matrix(coords)

st.sidebar.markdown(f"Number of cities: **{n_cities}**")

# Display parameters summary
st.markdown("### Selected Parameters")
st.write({
    "population": pop_size,
    "generations": generations,
    "elite_size": elite_size,
    "mutation_rate": mutation_rate,
    "tournament_k": tournament_k,
    "cities": n_cities,
    "random_seed": random_seed
})

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Route Map (interactive)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers+text', text=df['name'],
                             textposition='top center', marker=dict(size=8, color='blue')))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(width=2, color='red')))
    fig.update_layout(width=900, height=650, margin=dict(l=10, r=10, t=30, b=10))
    map_placeholder = st.plotly_chart(fig, use_container_width=True, key="map_initial")

with col2:
    st.subheader("Run Status")
    progress_text = st.empty()
    stats_placeholder = st.empty()
    best_table = st.empty()
    save_info = st.empty()

# Prepare population
if random_seed != 0:
    np.random.seed(int(random_seed))
population = init_population(n_cities, pop_size)

fitness_vals, lengths = fitness_population(population, dist_matrix)
best_idx = int(np.argmin(lengths))
best_route = population[best_idx]
best_length = float(lengths[best_idx])

st.session_state.setdefault("history_best", [])
st.session_state.setdefault("history_avg", [])

# Run GA when button pressed
if run_btn:
    history_best = []
    history_avg = []
    pop = population.copy()
    run_start = time.time()
    timestamp = now_timestamp()

    # params to save
    params = {
        "population": int(pop_size),
        "generations": int(generations),
        "elite_size": int(elite_size),
        "mutation_rate": float(mutation_rate),
        "tournament_k": int(tournament_k),
        "random_seed": int(random_seed),
        "cities": int(n_cities),
        "timestamp": timestamp
    }

    for gen in range(1, int(generations) + 1):
        pop, _, _ = evolve(pop, dist_matrix, elite_size=elite_size,
                           mutation_rate=mutation_rate, tournament_k=tournament_k)

        fitness_vals, lengths = fitness_population(pop, dist_matrix)

        avg_len = float(lengths.mean())
        best_len = float(lengths.min())
        best_idx = int(np.argmin(lengths))
        best_route = pop[best_idx]

        history_best.append(best_len)
        history_avg.append(avg_len)

        # update status
        progress_text.markdown(
            f"Generation {gen}/{generations} — Best: **{best_len:.4f}** — Avg: **{avg_len:.4f}**"
        )

        # update route figure (Plotly)
        xs = list(df.loc[best_route, 'x']) + [df.loc[best_route[0], 'x']]
        ys = list(df.loc[best_route, 'y']) + [df.loc[best_route[0], 'y']]
        fig.data[1].x = xs
        fig.data[1].y = ys

        # render updated map with unique key per gen to avoid DuplicateElementId
        map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_gen_{gen}")

        # show stats periodically
        if gen % max(1, int(generations) // 50) == 0 or gen == int(generations):
            stats_fig = go.Figure()
            stats_fig.add_trace(go.Scatter(y=history_best, mode='lines', name='best'))
            stats_fig.add_trace(go.Scatter(y=history_avg, mode='lines', name='avg'))
            stats_fig.update_layout(title="Fitness over Generations", xaxis_title="Generation", yaxis_title="Distance")
            stats_placeholder.plotly_chart(stats_fig, use_container_width=True, key=f"stats_gen_{gen}")

        # small throttle so UI stays responsive
        time.sleep(0.01)

    run_time = time.time() - run_start

    # final best route -> DataFrame
    route_ids = list(best_route)
    df_route = pd.DataFrame({
        "order": list(range(1, len(route_ids) + 1)),
        "city_id": route_ids,
        "name": df.loc[route_ids, 'name'].values,
        "x": df.loc[route_ids, 'x'].values,
        "y": df.loc[route_ids, 'y'].values
    })

    # Save run results
    out_folder = save_run_results(timestamp, params, df_route, history_best, history_avg, fig)

    # update session history
    st.session_state.history_best = history_best
    st.session_state.history_avg = history_avg

    st.success(f"Run completed in {run_time:.2f}s — best: {history_best[-1]:.4f}")
    save_info.markdown(f"Run saved to `{out_folder}`")

    # present final table and download
    best_table.dataframe(df_route)
    csv_bytes = df_route.to_csv(index=False).encode("utf-8")
    st.download_button("Download best route (CSV)", data=csv_bytes, file_name=f"best_route_{timestamp}.csv", mime="text/csv")

    # Generate aggregated professional summary after saving
    summary_info = generate_professional_summary()
    st.markdown("### Summary generated for all runs")
    st.write(f"Total runs aggregated: {summary_info['runs_count']}")
    st.write(f"Summary CSV: `{summary_info['summary_csv']}`")
    st.write(f"Aggregated fitness image: `{summary_info['fitness_all_runs_png']}`")
    st.write(f"Boxplot image: `{summary_info['boxplot_png']}`")
    if summary_info["best_run_info"] is not None:
        st.write("Best run (global):")
        st.json(summary_info["best_run_info"])

else:
    st.info("Press 'Run Algorithm' in the sidebar to execute a logged run.")
    # show initial map with a single default random route (for display)
    xs = list(df.loc[best_route, 'x']) + [df.loc[best_route[0], 'x']]
    ys = list(df.loc[best_route, 'y']) + [df.loc[best_route[0], 'y']]
    fig.data[1].x = xs
    fig.data[1].y = ys
    map_placeholder.plotly_chart(fig, use_container_width=True, key="map_initial_best")

    st.write("Initial sample best route (from random population):")
    st.write(f"Length: {best_length:.4f}")
    st.write(pd.DataFrame({
        "order": list(range(1, len(best_route) + 1)),
        "city_id": best_route,
        "name": df.loc[best_route, 'name'].values,
        "x": df.loc[best_route, 'x'].values,
        "y": df.loc[best_route, 'y'].values
    }))

# Quick dataset generator (keeps UX simple)
st.markdown("---")
if st.button("Generate Random Dataset (25 cities) & reload"):
    n = 25
    rng = np.random.default_rng()
    xs = rng.integers(0, 300, size=n)
    ys = rng.integers(0, 300, size=n)
    names = [f"C{i}" for i in range(n)]
    df_rand = pd.DataFrame({"id": list(range(n)), "name": names, "x": xs, "y": ys})
    st.session_state["rand_df"] = df_rand
    st.experimental_rerun()
