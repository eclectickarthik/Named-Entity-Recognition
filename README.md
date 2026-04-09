# Named Entity Recognition (NER) Demo 

Natural Language Processing (NLP) - 21CSE356T [Semester 6] - Team 5

A Named Entity Recognition (NER) web app built with BERT and Streamlit. Identifies and highlights entities like persons, organizations, locations, and miscellaneous terms in any text — with confidence scores, uncertainty detection, and analysis history.

---

## Tech Stack

- **Model**: `dslim/bert-base-NER` (BERT fine-tuned on CoNLL-2003)
- **Framework**: Streamlit
- **Library**: HuggingFace Transformers
- **Language**: Python 3.9

---

## Features

- Real-time NER with color-coded entity highlighting
- Confidence score with progress bar per entity
- Uncertainty detection — reduces confidence when hedging words like "maybe", "possibly" are detected
- Stats dashboard — total entities, persons, orgs, locations, avg confidence
- Past analyses history panel (session-based)

---

## Entity Types

| Tag | Type           | Color  |
|-----|----------------|--------|
| PER | Person         | Blue   |
| ORG | Organization   | Green  |
| LOC | Location       | Orange |
| MISC | Miscellaneous | Pink   |

---

## Requirements

```
streamlit
transformers
torch
numpy
```

---

## Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/eclectickarthik/Named-Entity-Recognition.git
cd ner-demo
```

**2. Install dependencies**
```bash
pip install streamlit transformers torch numpy
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

> First run will download the BERT model (~440MB). This is cached locally after the first run.

---

## Project Structure

```
ner-demo/
├── app.py          # Main Streamlit app
├── README.md       # This file
└── .gitignore      # Git ignore rules
```

---

## Model Details

- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Fine-tuned on**: CoNLL-2003 dataset (news articles tagged with PER, ORG, LOC, MISC)
- **Tokenizer**: WordPiece with `aggregation_strategy="first"` for clean multi-token entity merging
- **Inference**: CPU-based, no GPU required

---

## Known Limitations

- Model accuracy depends on context — ambiguous entity names may be misclassified
- Confidence reduction for uncertainty words is a post-processing heuristic, not model-level
- History resets on page refresh (session-state only, no persistent storage)

---

## Author

Karthik Ganesan
