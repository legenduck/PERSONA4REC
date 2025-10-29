# Data Preparation

LLM-based data preparation pipeline for generating item personas and user profiles from raw review data.

## Directory Structure

```
data_preparation/
├── core.py                          # Core utilities and data processing functions
├── amazon_tasks.py                  # Amazon-specific task implementations
├── yelp_tasks.py                    # Yelp-specific task implementations
├── generate.py                      # Main LLM generation orchestrator
├── prepare_intermediate_data.py     # Intermediate data file generator
├── run.sh                           # Full pipeline execution script
├── config/
│   └── profile_info.json.template   # LLM API configuration template
└── prompts/
    ├── amazon/                      # Amazon dataset prompts
    │   ├── aspects.txt
    │   ├── summary.txt
    │   ├── persona.txt
    │   └── selection.txt
    └── yelp/                        # Yelp dataset prompts
        ├── aspects.txt
        ├── summary.txt
        ├── persona.txt
        └── selection.txt
```

## Usage

### Full Pipeline (Using run.sh)

```bash
# 1. Configure LLM API settings
cp config/profile_info.json.template config/profile_info.json
# Edit config/profile_info.json with your API keys

# 2. Edit run.sh to set DATASET and TASK variables
# DATASET: amazon | yelp
# TASK: extract_aspects | generate_summary | generate_personas | prepare_intermediate | select_personas

# 3. Run the script
bash run.sh
```

### Step-by-Step Execution

Edit `TASK` variable in `run.sh` and run for each stage:

```bash
# Step 1: Extract aspects from reviews
# Edit run.sh: DATASET=amazon, TASK=extract_aspects
bash run.sh

# Step 2: Generate item summaries
# Edit run.sh: TASK=generate_summary
bash run.sh

# Step 3: Generate item personas
# Edit run.sh: TASK=generate_personas
bash run.sh

# Step 4: Prepare intermediate data (no LLM calls)
# Edit run.sh: TASK=prepare_intermediate
bash run.sh

# Step 5: Select matching personas (LLM as Judge)
# Edit run.sh: TASK=select_personas
bash run.sh
```

Or set via environment variables:

```bash
DATASET=amazon TASK=extract_aspects bash run.sh
DATASET=amazon TASK=generate_summary bash run.sh
DATASET=amazon TASK=generate_personas bash run.sh
DATASET=amazon TASK=prepare_intermediate bash run.sh
DATASET=amazon TASK=select_personas bash run.sh
```

## Configuration

### LLM API Settings (`config/profile_info.json`)

```json
{
  "default": {
    "api_key": "your-openai-api-key-here"
  }
}
```

### Model Parameters

Model parameters are specified via command-line arguments:
- `--model_name`: OpenAI model name (e.g., `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`)
- `--temperature`: Sampling temperature (0.0-1.0)
- `--max_tokens`: Maximum tokens to generate

## Output Files

The pipeline generates the following standardized output files:

| File | Description |
|------|-------------|
| `1_aspects.jsonl` | Extracted aspects from reviews |
| `2_summary.jsonl` | Item summaries with metadata |
| `2.5_grouped.jsonl` | Aspects grouped with summaries |
| `3_personas.jsonl` | Generated item personas (5 per item) |
| `3.5_gt.jsonl` | Ground truth user-item interactions |
| `3.5_history.jsonl` | Full user interaction histories |
| `3.5_history_loo.jsonl` | Leave-One-Out histories for evaluation |
| `3.5_persona_with_summary.jsonl` | Personas with item summaries |
| `4_selected_personas.jsonl` | Persona-user matching results |

## Prompts

Prompts are customized for each dataset and task:
- **aspects**: Extract key aspects/features from reviews
- **summary**: Summarize item characteristics from aspects
- **persona**: Generate diverse user perspectives as personas
- **selection**: Match personas to user preferences based on history

Prompts can be customized by editing files in `prompts/{dataset}/{task}.txt`

## Notes

- All scripts expect standardized field names (`user_id`, `item_id`, etc.)
- LLM API calls may incur costs - monitor usage carefully
- Use `--num_sample` flag with `generate.py` to test on a subset before full run
- `prepare_intermediate_data.py` does not make LLM calls and can be run freely


