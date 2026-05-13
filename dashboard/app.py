"""PulseLab Streamlit dashboard — experiment design, analysis, and verdict."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pulselab.analyze.cuped import cuped_adjust
from pulselab.analyze.hte import segment_effects
from pulselab.analyze.msprt import MsprtStream
from pulselab.analyze.srm import srm_check
from pulselab.data.synth import generate_experiment
from pulselab.design.power import sample_size_for_proportions
from pulselab.validate.synth_aa import run_synth_aa

st.set_page_config(page_title="PulseLab", page_icon="📊", layout="wide")


def render_verdict(label: str, ok: bool, detail: str) -> None:
    color = "#0a7c2f" if ok else "#a8322a"
    icon = "✅" if ok else "⚠️"
    st.markdown(
        f"<div style='padding:10px 14px;border-left:4px solid {color};background:#f7f7f7;'>"
        f"<b>{icon} {label}</b> &nbsp; <span style='color:#444'>{detail}</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("📊 PulseLab")
    st.caption("A/B testing with always-valid sequential testing, CUPED, SRM detection, and causal HTE")

    with st.sidebar:
        st.header("Experiment Design")
        n_per_arm = st.number_input("Sample size per arm", 100, 100_000, 5_000, step=500)
        baseline = st.number_input("Baseline metric mean", 0.001, 100.0, 4.81)
        true_effect = st.slider("True treatment effect (absolute)", -1.0, 1.0, 0.15, 0.05)
        rho = st.slider("Pre-period correlation (ρ)", 0.0, 0.95, 0.7, 0.05)
        seed = st.number_input("Random seed", 0, 999, 42)

        st.divider()
        st.header("Stopping Rule")
        alpha = st.slider("α (false-positive rate)", 0.01, 0.20, 0.05, 0.01)
        tau2 = st.slider("mSPRT prior variance τ²", 0.1, 5.0, 1.0, 0.1)

    tab_design, tab_results, tab_validate = st.tabs(
        ["🎯 Design", "📈 Analyze", "🧪 Synthetic A/A Validation"]
    )

    with tab_design:
        st.subheader("Sample-size calculator (proportions)")
        col1, col2, col3 = st.columns(3)
        with col1:
            base_rate = st.number_input("Baseline rate", 0.001, 0.999, 0.05, 0.005, key="bp")
        with col2:
            lift = st.number_input("Absolute lift to detect", 0.001, 0.5, 0.005, 0.001, key="lift")
        with col3:
            power = st.slider("Power (1 − β)", 0.5, 0.99, 0.8, 0.05, key="power")
        try:
            result = sample_size_for_proportions(base_rate, lift, power=power, alpha=alpha)
            st.metric("Per-arm sample size", f"{result.per_arm_n:,}")
            st.caption(
                f"To detect a {lift:.3f} absolute lift on a {base_rate:.1%} baseline at "
                f"α={alpha:.2f}, power={power:.0%} — total {result.total_n:,}."
            )
        except ValueError as e:
            st.error(str(e))

    with tab_results:
        if st.button("Generate synthetic experiment + analyze", type="primary"):
            with st.spinner("Generating synthetic data and running analysis..."):
                exp = generate_experiment(
                    n_control=n_per_arm,
                    n_treatment=n_per_arm,
                    baseline_mean=baseline,
                    true_effect=true_effect,
                    pre_period_corr=rho,
                    seed=int(seed),
                )

                # SRM
                srm = srm_check([len(exp.control_outcome), len(exp.treatment_outcome)])

                # mSPRT snapshot at the end
                stream = MsprtStream(tau2=tau2)
                stream.observe_many(exp.control_outcome, exp.treatment_outcome)
                snap = stream.snapshot(alpha=alpha)
                assert snap is not None

                # CUPED
                cup = cuped_adjust(
                    exp.treatment_outcome,
                    exp.control_outcome,
                    exp.treatment_pre,
                    exp.control_pre,
                )

                # HTE per segment
                segs = sorted(set(exp.control_segment) | set(exp.treatment_segment))
                seg_inputs = {
                    s: (
                        exp.control_outcome[exp.control_segment == s],
                        exp.treatment_outcome[exp.treatment_segment == s],
                    )
                    for s in segs
                }
                hte = segment_effects(seg_inputs, q=alpha)

            st.subheader("Verdict")
            verdict_text = (
                "Reject H₀ — ship treatment"
                if snap.reject_null(alpha=alpha)
                else "Fail to reject H₀ — no significant effect yet"
            )
            render_verdict(
                f"mSPRT (always-valid)",
                snap.reject_null(alpha=alpha),
                f"effect={snap.mean_diff:+.4f}, p={snap.p_value:.4f}, "
                f"CI [{snap.ci_low:+.4f}, {snap.ci_high:+.4f}] — {verdict_text}",
            )
            render_verdict(
                "Sample Ratio Mismatch check",
                srm.healthy,
                srm.summary(),
            )

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Naive effect",
                f"{cup.naive_effect:+.4f}",
                help="Difference of arm means without CUPED adjustment",
            )
            col2.metric(
                "CUPED effect",
                f"{cup.cuped_effect:+.4f}",
                f"{cup.variance_reduction:.1%} CI shrinkage",
            )
            col3.metric(
                "True effect (oracle)",
                f"{true_effect:+.4f}",
                help="The ground-truth effect baked into the synthetic data generator",
            )

            st.divider()
            st.subheader("Heterogeneous Treatment Effects (per segment)")
            if hte:
                rows = [
                    {
                        "segment": e.segment,
                        "lift": e.effect,
                        "p_value": e.p_value,
                        "p_adjusted (BH)": e.p_adjusted,
                        "significant": "✅" if e.significant else "—",
                        "n_control": e.n_control,
                        "n_treatment": e.n_treatment,
                    }
                    for e in hte
                ]
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.info("No segments large enough to analyze.")

            st.divider()
            st.subheader("Distribution of outcomes")
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(x=exp.control_outcome, name="Control", opacity=0.7, nbinsx=40)
            )
            fig.add_trace(
                go.Histogram(x=exp.treatment_outcome, name="Treatment", opacity=0.7, nbinsx=40)
            )
            fig.update_layout(
                barmode="overlay",
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Metric value",
                yaxis_title="Users",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_validate:
        st.markdown(
            "Runs **N null A/A experiments with daily peeking** under the mSPRT "
            "stopping rule. If the math is right, empirical FPR ≤ α — peeking is safe."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            n_exp = st.number_input("Number of experiments", 100, 5_000, 500, step=100)
        with col2:
            per_day = st.number_input("Users per arm per day", 50, 1_000, 200, step=50)
        with col3:
            n_days = st.number_input("Max days per experiment", 5, 90, 30, step=5)
        if st.button("Run synthetic A/A", type="primary"):
            with st.spinner(f"Running {n_exp} bootstrapped null experiments..."):
                aa = run_synth_aa(
                    n_experiments=int(n_exp),
                    per_arm_per_day=int(per_day),
                    n_days=int(n_days),
                    alpha=alpha,
                    tau2=tau2,
                    seed=int(seed),
                )
            ok = aa.passed
            render_verdict(
                f"Empirical FPR = {aa.fpr:.3f} (target α = {aa.target_alpha:.2f})",
                ok,
                f"{aa.n_false_positives} false stops in {aa.n_experiments} A/A runs · "
                f"avg sample size at end {aa.avg_n_at_stop:,.0f}",
            )
            st.caption(
                "**Why this matters:** with a standard t-test, peeking daily inflates "
                "false-positive rate from ~5% to ~20–30%. mSPRT keeps it bounded by α."
            )


if __name__ == "__main__":
    main()
