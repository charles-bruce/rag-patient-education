# RAG-Based LLMs for Post-Discharge Patient Education

Code, prompts, test sets and run logs for the paper *RAG-Based LLMs for
Post-Discharge Patient Education: Evaluating Faithfulness and Readability*
(EHB 2026).

A retrieval-augmented patient education system over the English MedlinePlus
corpus, running entirely on local hardware, evaluated on 85 held-out questions
for grounding and for the effect of prompt-level reading-level control.

**The outputs in this repository have not been reviewed by clinicians and are
not intended for clinical use.**

## What is here

| | |
|---|---|
| `app.py` | Streamlit prototype. Same configuration as the evaluation. |
| `app_personalised.py` | Optional condition-profile variant. Implemented and demonstrable but **not evaluated**, and not reported in the paper. |
| `PROMPTS.md` | Every generation and judge prompt, verbatim. |
| `mplus_topics_2026-07-07.xml` | The corpus snapshot the results were produced from. |
| `wk7_readability_heldout.jsonl` | The held-out run log. Every number in the paper is computed from this file. |

### Pipeline

Run in order. Notebooks pass data through the JSON and JSONL files in this
directory; none imports another.

| Notebook | Does |
|---|---|
| `01_parse_clean.ipynb` | Parses the MedlinePlus XML, strips HTML, keeps English topics over 30 words → 1,016 topics |
| `02_eda.ipynb` | Corpus description |
| `03_build_retriever.ipynb` | Paragraph chunking, title + synonym prefixes, embedding, Chroma index → 10,337 chunks. Applies the maximum of two chunks per topic |
| `04_generator_eval.ipynb` | Generator selection, first judge calibration |
| `05_build_test_set.ipynb` | MedQuAD → 200 questions, topic-disjoint 115/85 split |
| `06_eval_harness.ipynb` | Metric implementations |
| `07_readability_dev.ipynb` | Reading-level experiment on the development set |
| `08_heldout_run.ipynb` | **The held-out run.** Opened once, at the end, with no configuration changed afterwards. Produces `wk7_readability_heldout.jsonl` |
| `09_aggregate_dev.ipynb` | Development-set aggregates |


### Data and results

`test_queries_dev.json` (115), `test_queries_heldout.json` (85),
`analysis_categories.json` (the seven categories, fixed before the run),
`gold_subset.json`, `personalisation_questions.json`.

Run logs `wk4_runs.jsonl`, `wk5_runs.jsonl`, `wk6_readability_dev.jsonl`,
`wk7_readability_heldout.jsonl`. Judge calibration in `judge_calibration*.csv`.

## Reproducing

```bash
pip install -r requirements.txt
ollama pull mistral:7b-instruct
ollama pull llama3.1:8b
```

Run the notebooks in numbered order. `01` through `03` build the index; `08` is
the held-out run. The Chroma index and the parsed topic JSON are
derived artefacts and are not committed — both rebuild from the XML snapshot.
The full held-out run took 248 minutes on a single machine.

`streamlit run app.py` for the prototype (needs the index built first).

## Configuration

Generator `mistral:7b-instruct`, 7.2B, Q4_K_M, via Ollama.
Judge `llama3.1:8b`, 8.0B, Q4_K_M — deliberately a different model from the
generator, since language models score their own output more generously.
Embeddings `BAAI/bge-small-en-v1.5`, cosine similarity, Chroma.
Five passages retrieved, at most two per topic. `temperature=0`, `seed=42`
throughout.

Faithfulness, relevancy and context precision are reimplementations of the
RAGAS definitions rather than calls to the RAGAS library, so that each
judgement is a separate inspectable model call. Readability uses `textstat`;
answers under ten words are excluded because the formulas are unstable below
that length. Citation precision is an exact string match between cited topic
titles and the retrieved set.

## Data sources

MedlinePlus health topics XML, US National Library of Medicine —
<https://medlineplus.gov/xml.html>, snapshot of 7 July 2026, included here so
the results can be reproduced exactly.

Test questions derive from MedQuAD (Ben Abacha and Demner-Fushman, 2019),
CC BY 4.0. MedQuAD itself is **not** redistributed here; the question sets in
this repository are the derived subsets. Get MedQuAD from
<https://github.com/abachaa/MedQuAD>.

## Limitations

The judge model was never validated against human raters, so absolute metric
values are uncalibrated; comparisons between arms use the same judge and are
more secure. No clinician or patient assessed any output. Refusal is emergent
from the prompt rather than enforced. A single generator was used, so
prompt-level and model-level explanations cannot be separated.
