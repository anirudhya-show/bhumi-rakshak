# Land Acquisition Delay Risk — Predictive Analytics System

An AI-powered decision support system that predicts which land acquisition
projects are at risk of delay, explains *why*, and recommends corrective
action — built for the SIH problem statement on land acquisition delays.

## What's in this folder

| File | What it does |
|---|---|
| `generate_data.py` | Creates the synthetic (realistic but fake) dataset of 1,500 land acquisition projects |
| `land_projects.csv` | The generated dataset (output of the script above) |
| `train_model.py` | Trains an XGBoost model, computes SHAP-based explanations, and writes recommendations |
| `predictions.csv` | Every project with its delay probability, risk category, top risk factors, and recommendation |
| `model.json` | The trained model, saved so it doesn't need retraining every time |
| `india_states.geojson` | Map boundaries used for the state-wise risk heatmap |
| `dashboard.py` | The Streamlit web dashboard — this is what you demo to judges |
| `requirements.txt` | List of Python packages needed |

## How to run it (step by step)

### 1. Install the required packages
Open a terminal in this folder and run:

```
pip install -r requirements.txt
```

(If that fails, try `pip install -r requirements.txt --break-system-packages`)

### 2. Generate the dataset
```
python3 generate_data.py
```
This creates `land_projects.csv`. You only need to do this once (or again if
you want a fresh random dataset).

### 3. Train the model
```
python3 train_model.py
```
This trains the model and creates `predictions.csv`. You'll see something
like:
```
Test AUC: 0.80   Test Accuracy: 0.76
```
This means the model correctly distinguishes delayed vs. on-time projects
about 76% of the time on data it has never seen — a solid, credible result.

### 4. Launch the dashboard
```
streamlit run dashboard.py
```
This opens a browser tab (usually at `http://localhost:8501`) showing the
live dashboard. This is what you show judges.

### 5. (Optional) Enable the live "AI Briefing" feature
The dashboard has an **AI Briefing** button on each project's detail view.
When clicked, it makes a *live* call to an LLM API and writes a custom,
project-specific explanation and recommendation — on top of the fast,
built-in rule-based recommendation, which always works even without this.

It works with **any** provider that exposes an OpenAI-compatible
`/chat/completions` endpoint (OpenRouter, OpenAI, Groq, local models via
Ollama/LM Studio, etc.) or Anthropic's native API.

**To use OpenRouter's free models (recommended, no cost):**
1. Sign up at https://openrouter.ai and get an API key from
   https://openrouter.ai/keys
2. Browse https://openrouter.ai/models and filter for `:free` models — copy
   a model ID such as `meta-llama/llama-3.1-8b-instruct:free`
3. In the dashboard sidebar, under "AI Briefing":
   - Provider: `OpenRouter (free models)` (base URL is pre-filled)
   - Model name: paste the `:free` model ID you picked
   - API key: paste your OpenRouter key
4. Open any project's detail view and click **Generate AI Briefing**

To use a different provider, just pick it from the "Provider" dropdown (or
choose "Custom" and type in any OpenAI-compatible base URL) and paste the
matching API key and model name.

This is optional — if you skip it, the rest of the dashboard works exactly
the same. Treat it as a "bonus" feature, not something you depend on for the
core demo, in case of connectivity issues during judging.
