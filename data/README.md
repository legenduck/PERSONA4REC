# Data Directory
This directory contains raw and processed data for the Persona4Rec framework.
## Directory Structure
```
data/
├── raw/                    # Raw data (metadata and reviews)
│   ├── amazon/
│   │   ├── meta_amazon.jsonl
│   │   └── review_amazon.jsonl
│   └── yelp/
│       ├── meta_yelp.jsonl
│       └── review_yelp.jsonl
├── processed/              # Processed data (training data)
│   ├── amazon/
│   │   ├── 1_aspects.jsonl
│   │   ├── 2_summary.jsonl
│   │   ├── 2.5_grouped.jsonl
│   │   ├── 3_personas.jsonl
│   │   ├── 3.5_gt.jsonl
│   │   ├── 3.5_history.jsonl
│   │   ├── 3.5_history_loo.jsonl
│   │   ├── 3.5_persona_with_summary.jsonl
│   │   └── 4_selected_personas.jsonl
│   └── yelp/
│       ├── 1_aspects.jsonl
│       ├── 2_summary.jsonl
│       ├── 2.5_grouped.jsonl
│       ├── 3_personas.jsonl
│       ├── 3.5_gt.jsonl
│       ├── 3.5_history.jsonl
│       ├── 3.5_history_loo.jsonl
│       ├── 3.5_persona_with_summary.jsonl
│       └── 4_selected_personas.jsonl
└── sample/                 # Sample data (50 lines)
    ├── amazon/
    │   ├── 1_aspects.jsonl
    │   ├── 2_summary.jsonl
    │   ├── 2.5_grouped.jsonl
    │   ├── 3_personas.jsonl
    │   ├── 3.5_gt.jsonl
    │   ├── 3.5_history.jsonl
    │   ├── 3.5_history_loo.jsonl
    │   ├── 3.5_persona_with_summary.jsonl
    │   └── 4_selected_personas.jsonl
    └── yelp/
        ├── 1_aspects.jsonl
        ├── 2_summary.jsonl
        ├── 2.5_grouped.jsonl
        ├── 3_personas.jsonl
        ├── 3.5_gt.jsonl
        ├── 3.5_history.jsonl
        ├── 3.5_history_loo.jsonl
        ├── 3.5_persona_with_summary.jsonl
        └── 4_selected_personas.jsonl
```
## File Descriptions
### Raw Data (`raw/`)
* `meta_{dataset}.jsonl` - Item metadata (e.g., `meta_amazon.jsonl`)
* `review_{dataset}.jsonl` - User reviews (e.g., `review_amazon.jsonl`)
### Processed Data (`processed/{dataset}/`)
1. **1_aspects.jsonl**: Extracted aspects from reviews
2. **2_summary.jsonl**: Item summaries
3. **2.5_grouped.jsonl**: Grouped aspects with summaries
4. **3_personas.jsonl**: Generated item personas
5. **3.5_gt.jsonl**: Ground truth interaction data
6. **3.5_history.jsonl**: User history data
7. **3.5_history_loo.jsonl**: Leave-One-Out history for evaluation
8. **3.5_persona_with_summary.jsonl**: Personas combined with summaries
9. **4_selected_personas.jsonl**: Selected persona matching information
### Sample Data (`sample/{dataset}/`)
Due to Git repository size limitations, the full processed data files are not included in this repository. Instead, we provide sample files containing 50 lines from each processed file to demonstrate the data structure and format