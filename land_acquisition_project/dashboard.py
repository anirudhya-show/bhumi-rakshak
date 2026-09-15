"""
dashboard.py
-------------
The interactive web dashboard. Run with:

    streamlit run dashboard.py

It reads predictions.csv (produced by train_model.py) and shows:
  1. An overview: KPIs + risk distribution + state-wise map
  2. A sortable/filterable table of all projects
  3. A detail view for any single project with its SHAP-based explanation
"""

import json
import os
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Land Acquisition Delay Risk Dashboard", layout="wide")

# Map our dataset's state names to the names used in the GeoJSON file
STATE_NAME_FIX = {
    "Odisha": "Orissa",
    "Uttarakhand": "Uttaranchal",
}

# Presets for the AI Briefing feature. "openai" style = the widely-used
# /chat/completions format (works for OpenRouter, OpenAI, Groq, Together,
# local servers like Ollama/LM Studio, etc). "anthropic" style = Claude's
# native /messages format.
PROVIDER_PRESETS = {
    "OpenRouter (free models)": {
        "style": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "example_model": "meta-llama/llama-3.1-8b-instruct:free",
    },
    "OpenAI": {
        "style": "openai",
        "base_url": "https://api.openai.com/v1",
        "example_model": "gpt-4o-mini",
    },
    "Groq": {
        "style": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "example_model": "llama-3.1-8b-instant",
    },
    "Anthropic": {
        "style": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "example_model": "claude-sonnet-5",
    },
    "Custom (OpenAI-compatible)": {
        "style": "openai",
        "base_url": "",
        "example_model": "",
    },
}


def get_bbox(geojson_data, geo_names=None):
    """Compute (min_lon, max_lon, min_lat, max_lat) for the given state names
    in the geojson (or for ALL features if geo_names is None)."""
    lons, lats = [], []

    def walk(coords):
        # Recursively descend until we hit [lon, lat] pairs
        if isinstance(coords[0], (int, float)):
            lons.append(coords[0])
            lats.append(coords[1])
        else:
            for c in coords:
                walk(c)

    for feat in geojson_data["features"]:
        name = feat["properties"].get("NAME_1")
        if geo_names is not None and name not in geo_names:
            continue
        walk(feat["geometry"]["coordinates"])

    if not lons:
        return None
    return min(lons), max(lons), min(lats), max(lats)


def build_briefing_prompt(row, factors):
    factor_lines = "\n".join(f"- {f}" for f in factors) if factors else "- No major risk-increasing factors identified"
    return f"""You are an infrastructure project risk analyst. Write a short (120-150 word) briefing
for a government official about this land acquisition project. Be direct and practical.

Project ID: {row['project_id']}
State: {row['state']}
Project type: {row['project_type']}
Predicted delay probability: {row['delay_probability']:.0%}
Risk category: {row['risk_category']}
Top risk-increasing factors (from the ML model's explanation):
{factor_lines}

Raw project data:
- Land area: {row['land_area_hectares']} hectares
- Affected families: {row['affected_families']}
- Compensation disbursed: {row['compensation_disbursed_pct']}%
- Days in approval stage: {row['approval_stage_days']}
- Legal disputes: {row['legal_disputes_count']}
- Rehabilitation progress: {row['rehabilitation_progress_pct']}%
- Documentation completeness: {row['documentation_completeness_pct']}%
- Stakeholder responsiveness score (1-10): {row['stakeholder_responsiveness_score']}
- Department's historical delay rate: {row['past_dept_delay_rate_pct']}%

Write the briefing as flowing prose (no headers/bullets), covering: (1) the situation in plain
terms, (2) the most likely cause of delay, (3) one concrete, specific recommended next step."""


def call_ai_briefing(style, base_url, model, api_key, row, factors):
    """Live call to an LLM API to generate a custom natural-language briefing
    for one project. Works with any OpenAI-compatible endpoint (OpenRouter,
    OpenAI, Groq, local servers, etc.) or Anthropic's native format."""
    prompt = build_briefing_prompt(row, factors)
    base_url = base_url.rstrip("/")

    if style == "anthropic":
        resp = requests.post(
            f"{base_url}/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []))

    # Default: OpenAI-compatible /chat/completions format
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


@st.cache_data
def load_data():
    df = pd.read_csv("predictions.csv")
    return df


@st.cache_data
def load_geojson():
    with open("india_states.geojson") as f:
        return json.load(f)


df = load_data()
geojson = load_geojson()

st.title("🏗️ Land Acquisition Delay — Predictive Risk Dashboard")
st.caption(
    "Predicts which land acquisition projects are at risk of delay, explains WHY, "
    "and recommends corrective action. Built on synthetic sample data for demonstration."
)

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
states = st.sidebar.multiselect("State", sorted(df["state"].unique()))
ptypes = st.sidebar.multiselect("Project type", sorted(df["project_type"].unique()))
risk_levels = st.sidebar.multiselect(
    "Risk category", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
)

st.sidebar.divider()
st.sidebar.header("AI Briefing (optional)")
st.sidebar.caption(
    "Enables a live, project-specific AI-written briefing. Works with any "
    "OpenAI-compatible API (OpenRouter, OpenAI, Groq, local models, etc.) or Anthropic."
)
provider_name = st.sidebar.selectbox("Provider", list(PROVIDER_PRESETS.keys()))
preset = PROVIDER_PRESETS[provider_name]

base_url_input = st.sidebar.text_input("API base URL", value=preset["base_url"])
model_input = st.sidebar.text_input(
    "Model name", value=preset["example_model"],
    help="For OpenRouter's free tier, check openrouter.ai/models for current ':free' model IDs.",
)
default_key = os.environ.get("LLM_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "API key", value=default_key, type="password", placeholder="sk-..."
)
st.sidebar.caption("Your key is only kept for this browser session — never saved to disk.")

filtered = df.copy()
if states:
    filtered = filtered[filtered["state"].isin(states)]
if ptypes:
    filtered = filtered[filtered["project_type"].isin(ptypes)]
if risk_levels:
    filtered = filtered[filtered["risk_category"].isin(risk_levels)]

# ---------------- KPI row ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total projects", len(filtered))
col2.metric("High risk", int((filtered["risk_category"] == "High").sum()))
col3.metric("Medium risk", int((filtered["risk_category"] == "Medium").sum()))
col4.metric("Avg. delay probability", f"{filtered['delay_probability'].mean():.0%}" if len(filtered) else "—")

st.divider()

# ---------------- Map + risk distribution ----------------
map_col, chart_col = st.columns([1.3, 1])

with map_col:
    st.subheader("State-wise average delay risk")
    if states:
        st.caption(f"Zoomed to: {', '.join(states)}")
    else:
        st.caption("Showing all of India. Select states in the sidebar to zoom in.")

    all_geo_states = sorted({f["properties"].get("NAME_1") for f in geojson["features"]})

    # Risk coloring respects the project-type / risk-category filters, but NOT
    # the state selection itself — state selection only controls zoom/highlight.
    map_source = df.copy()
    if ptypes:
        map_source = map_source[map_source["project_type"].isin(ptypes)]
    if risk_levels:
        map_source = map_source[map_source["risk_category"].isin(risk_levels)]

    state_avg = map_source.groupby("state")["delay_probability"].mean().reset_index()
    state_avg["state_geo"] = state_avg["state"].replace(STATE_NAME_FIX)

    # If the user picked specific states, only highlight those in the foreground layer.
    if states:
        geo_selected = [STATE_NAME_FIX.get(s, s) for s in states]
        highlight_df = state_avg[state_avg["state_geo"].isin(geo_selected)]
    else:
        highlight_df = state_avg
        geo_selected = None

    fig_map = go.Figure()

    # Background layer: full India outline in neutral grey, always fully visible.
    fig_map.add_trace(go.Choropleth(
        geojson=geojson,
        featureidkey="properties.NAME_1",
        locations=all_geo_states,
        z=[0] * len(all_geo_states),
        colorscale=[[0, "#eef0f2"], [1, "#eef0f2"]],
        showscale=False,
        marker_line_color="white",
        marker_line_width=0.6,
        hoverinfo="skip",
    ))

    # Foreground layer: risk-colored states (all of them, or just the selected ones).
    if len(highlight_df):
        fig_map.add_trace(go.Choropleth(
            geojson=geojson,
            featureidkey="properties.NAME_1",
            locations=highlight_df["state_geo"],
            z=highlight_df["delay_probability"],
            zmin=0, zmax=1,
            colorscale="Reds",
            marker_line_color="white",
            marker_line_width=0.8,
            colorbar_title="Avg. delay risk",
            text=highlight_df["state"],
            customdata=highlight_df["delay_probability"],
            hovertemplate="<b>%{text}</b><br>Avg. delay probability: %{customdata:.0%}<extra></extra>",
        ))

    # Zoom: full India when nothing selected, tight bounding box around the
    # selected state(s) otherwise.
    bbox = get_bbox(geojson, geo_names=geo_selected)
    if bbox:
        min_lon, max_lon, min_lat, max_lat = bbox
        pad_lon = max((max_lon - min_lon) * 0.15, 0.5)
        pad_lat = max((max_lat - min_lat) * 0.15, 0.5)
        fig_map.update_geos(
            visible=False,
            lonaxis_range=[min_lon - pad_lon, max_lon + pad_lon],
            lataxis_range=[min_lat - pad_lat, max_lat + pad_lat],
        )
    else:
        fig_map.update_geos(visible=False, fitbounds="geojson")

    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
    st.plotly_chart(fig_map, width='stretch')

with chart_col:
    st.subheader("Risk category breakdown")
    cat_counts = filtered["risk_category"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    fig_bar = px.bar(
        x=cat_counts.index, y=cat_counts.values,
        color=cat_counts.index,
        color_discrete_map={"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"},
        labels={"x": "Risk category", "y": "Number of projects"},
    )
    fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=420)
    st.plotly_chart(fig_bar, width='stretch')

st.divider()

# ---------------- Project table ----------------
st.subheader("Projects (sorted by risk)")
display_cols = ["project_id", "state", "project_type", "delay_probability", "risk_category", "top_factor_1"]
table = filtered.sort_values("delay_probability", ascending=False)[display_cols].rename(
    columns={
        "project_id": "Project ID",
        "state": "State",
        "project_type": "Type",
        "delay_probability": "Delay probability",
        "risk_category": "Risk",
        "top_factor_1": "Top risk driver",
    }
)
st.dataframe(
    table.style.format({"Delay probability": "{:.0%}"}),
    width='stretch',
    height=350,
)

st.divider()

# ---------------- Project detail / explanation ----------------
st.subheader("🔍 Project detail & explanation")
selected_id = st.selectbox("Select a project to inspect", filtered["project_id"].tolist())

if selected_id:
    row = df[df["project_id"] == selected_id].iloc[0]

    d1, d2 = st.columns([1, 1.4])
    with d1:
        st.markdown(f"### {row['project_id']} — {row['state']}")
        st.markdown(f"**Type:** {row['project_type']}")
        risk_color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[row["risk_category"]]
        st.markdown(f"**Risk category:** {risk_color} {row['risk_category']}")
        st.markdown(f"**Delay probability:** {row['delay_probability']:.0%}")
        st.progress(float(row["delay_probability"]))

        st.markdown("#### Raw project data")
        raw_fields = [
            "land_area_hectares", "affected_families", "compensation_disbursed_pct",
            "approval_stage_days", "legal_disputes_count", "rehabilitation_progress_pct",
            "documentation_completeness_pct", "stakeholder_responsiveness_score",
            "past_dept_delay_rate_pct",
        ]
        st.table(pd.DataFrame({"Field": raw_fields, "Value": [row[f] for f in raw_fields]}))

    with d2:
        st.markdown("#### Why this risk score? (top contributing factors)")
        factors = [row["top_factor_1"], row["top_factor_2"], row["top_factor_3"]]
        factors = [f for f in factors if isinstance(f, str) and f.strip()]
        if factors:
            for i, f in enumerate(factors, 1):
                st.markdown(f"**{i}. {f}**")
        else:
            st.markdown("_No strong risk-increasing factors identified — this project looks low-risk._")

        st.markdown("#### Recommended action")
        rec = row.get("top_factor_1_recommendation", "")
        if isinstance(rec, str) and rec.strip():
            st.success(rec)
        else:
            st.info("No corrective action needed at this time — continue routine monitoring.")

        st.markdown("#### 🧠 AI Briefing (live)")
        st.caption("Calls the Claude API in real time to write a custom briefing for this specific project.")

        briefing_key = f"briefing_{selected_id}"
        generate_clicked = st.button("Generate AI Briefing", key=f"btn_{selected_id}")

        if generate_clicked:
            if not api_key_input:
                st.warning("Enter your API key in the sidebar first.")
            elif not base_url_input or not model_input:
                st.warning("Fill in the API base URL and model name in the sidebar.")
            else:
                with st.spinner(f"Calling {provider_name}..."):
                    try:
                        briefing_text = call_ai_briefing(
                            preset["style"], base_url_input, model_input, api_key_input, row, factors
                        )
                        st.session_state[briefing_key] = briefing_text
                    except requests.exceptions.HTTPError as e:
                        st.error(f"API returned an error: {e.response.status_code} — {e.response.text[:300]}")
                    except Exception as e:
                        st.error(f"AI Briefing failed: {e}")

        if briefing_key in st.session_state:
            st.markdown(st.session_state[briefing_key])

st.divider()
st.caption(
    "⚠️ This dashboard runs on synthetically generated demonstration data built to reflect "
    "realistic parameter relationships described in the problem statement. It is not connected "
    "to live government land records."
)
