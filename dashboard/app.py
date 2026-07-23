"""PulseLab Streamlit dashboard - experiment design, analysis, and verdict."""
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

st.set_page_config(page_title="PulseLab", layout="wide")

# ---- Theme: navy / teal / gold, Poppins + Inter, to match the PulseLab site ----
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
:root{--navy:#1a2b4a;--teal:#3d9b8c;--teal-light:#5ec4b0;--gold:#e6c15a;--muted:#6b7a90;}
.stApp{background:linear-gradient(160deg,#eef3f5 0%,#e8eef1 100%);}
html, body, [class*="css"]{font-family:'Inter',sans-serif;color:#1a2b4a;}
h1,h2,h3,h4{font-family:'Poppins',sans-serif !important;letter-spacing:-.5px;color:#1a2b4a;}
section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #d9e2e6;}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  font-size:14px !important;text-transform:uppercase;letter-spacing:.8px;color:#1a2b4a;}
.stButton>button{background:#1a2b4a;color:#fff;border:none;border-radius:9px;
  font-weight:600;padding:10px 22px;font-family:'Inter',sans-serif;transition:transform .15s;}
.stButton>button:hover{background:#26395c;color:#fff;transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid #d9e2e6;}
.stTabs [data-baseweb="tab"]{font-weight:600;color:#6b7a90;}
.stTabs [aria-selected="true"]{color:#3d9b8c;}
.stTabs [data-baseweb="tab-highlight"]{background:#3d9b8c;}
[data-testid="stMetricValue"]{font-family:'Poppins',sans-serif;color:#1a2b4a;}
[data-testid="stMetricDelta"]{color:#2f8f7f;}
input[type=range]{accent-color:#3d9b8c;}
.pl-brand{display:flex;align-items:center;gap:12px;margin-bottom:2px;}
.pl-mark{width:40px;height:40px;border-radius:10px;background:#1a2b4a;display:flex;
  align-items:center;justify-content:center;}
.pl-name{font-family:'Poppins',sans-serif;font-weight:700;font-size:34px;letter-spacing:-1px;color:#1a2b4a;}
.pl-badge{display:inline-block;font-family:'Poppins',sans-serif;font-weight:600;font-size:12px;
  padding:3px 10px;border-radius:20px;margin-right:8px;}
</style>
"""

PL_HEADER = """
<div class="pl-brand">
  <span class="pl-mark"><svg width="22" height="22" viewBox="0 0 24 24" fill="none"
    stroke="#5ec4b0" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12h4l2 6 4-14 2 8h6"/></svg></span>
  <span class="pl-name">PulseLab</span>
</div>
"""

WALKTHROUGH = """
<div style="background:#fff;border:1px solid #d9e2e6;border-radius:12px;
  padding:20px 24px;margin:14px 0 26px;box-shadow:0 8px 24px rgba(26,43,74,.06);">
  <div style="font-family:'Poppins',sans-serif;font-weight:700;font-size:16px;
    color:#1a2b4a;margin-bottom:14px;">How to use PulseLab</div>
  <div style="display:flex;gap:18px;flex-wrap:wrap;">
    <div style="flex:1;min-width:190px;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:6px;">
        <span style="width:24px;height:24px;border-radius:7px;background:#1a2b4a;color:#fff;
          font-family:'Poppins';font-weight:700;font-size:13px;display:flex;
          align-items:center;justify-content:center;">1</span>
        <b style="color:#1a2b4a;">Set parameters</b></div>
      <div style="font-size:13.5px;color:#6b7a90;line-height:1.55;">
        Use the sidebar on the left to set your sample size, baseline, effect, and stopping rule.</div>
    </div>
    <div style="flex:1;min-width:190px;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:6px;">
        <span style="width:24px;height:24px;border-radius:7px;background:#3d9b8c;color:#fff;
          font-family:'Poppins';font-weight:700;font-size:13px;display:flex;
          align-items:center;justify-content:center;">2</span>
        <b style="color:#1a2b4a;">Design and analyze</b></div>
      <div style="font-size:13.5px;color:#6b7a90;line-height:1.55;">
        Size the test on the Design tab, then run a synthetic experiment on the Analyze tab
        to see the mSPRT verdict, CUPED effect, SRM check, and per-segment lift.</div>
    </div>
    <div style="flex:1;min-width:190px;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:6px;">
        <span style="width:24px;height:24px;border-radius:7px;background:#e6c15a;color:#1a2b4a;
          font-family:'Poppins';font-weight:700;font-size:13px;display:flex;
          align-items:center;justify-content:center;">3</span>
        <b style="color:#1a2b4a;">Prove it is peek-safe</b></div>
      <div style="font-size:13.5px;color:#6b7a90;line-height:1.55;">
        On the Synthetic A/A tab, run many null experiments with daily peeking and confirm
        the empirical false-positive rate stays at or below your alpha.</div>
    </div>
  </div>
</div>
"""


def render_verdict(label: str, ok: bool, detail: str) -> None:
    accent = "#3d9b8c" if ok else "#c98a2a"
    bg = "#f2f8f6" if ok else "#fbf6ec"
    tag = "PASS" if ok else "CHECK"
    st.markdown(
        f"<div style='padding:12px 16px;border-left:4px solid {accent};background:{bg};"
        f"border-radius:8px;margin-bottom:12px;'>"
        f"<span class='pl-badge' style='background:{accent};color:#fff;'>{tag}</span>"
        f"<b>{label}</b> &nbsp; <span style='color:#41506b'>{detail}</span></div>",
        unsafe_allow_html=True,
    )


SITE_URL = "http://localhost:8000"

BACK_LINK = f"""
<a href="{SITE_URL}" style="display:inline-flex;align-items:center;gap:7px;
  font-size:14px;font-weight:600;color:#1a2b4a;text-decoration:none;
  border:1px solid #d9e2e6;background:#fff;border-radius:9px;padding:8px 14px;
  margin-bottom:14px;">
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3d9b8c"
    stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M19 12H5M12 19l-7-7 7-7"/></svg>
  Back to home
</a>
"""


def main() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(BACK_LINK, unsafe_allow_html=True)
    st.markdown(PL_HEADER, unsafe_allow_html=True)
    st.caption("A/B testing with always-valid sequential testing, CUPED, SRM detection, and causal HTE")
    st.markdown(WALKTHROUGH, unsafe_allow_html=True)

    with st.sidebar:
        st.caption("Set your experiment parameters here, then use the tabs on the right.")
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
        ["Design", "Analyze", "Synthetic A/A Validation"]
    )

    with tab_design:
        st.caption("Step 1: enter your baseline rate and the lift you want to detect. "
                   "PulseLab returns how many users per arm you need.")
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
                f"α={alpha:.2f}, power={power:.0%} - total {result.total_n:,}."
            )
        except ValueError as e:
            st.error(str(e))

    with tab_results:
        st.caption("Step 2: click the button to generate a synthetic experiment from your "
                   "sidebar settings and read the verdict, CUPED effect, SRM check, and segment lift.")
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
                "Reject H₀ - ship treatment"
                if snap.reject_null(alpha=alpha)
                else "Fail to reject H₀ - no significant effect yet"
            )
            render_verdict(
                f"mSPRT (always-valid)",
                snap.reject_null(alpha=alpha),
                f"effect={snap.mean_diff:+.4f}, p={snap.p_value:.4f}, "
                f"CI [{snap.ci_low:+.4f}, {snap.ci_high:+.4f}] - {verdict_text}",
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
                        "significant": "yes" if e.significant else "no",
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
        st.caption("Step 3: set the number of null experiments and click Run to prove that "
                   "peeking stays safe under the mSPRT stopping rule.")
        st.markdown(
            "Runs **N null A/A experiments with daily peeking** under the mSPRT "
            "stopping rule. If the math is right, empirical FPR ≤ α - peeking is safe."
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
                "false-positive rate from about 5% to 20 or 30%. mSPRT keeps it bounded by α."
            )


if __name__ == "__main__":
    main()
