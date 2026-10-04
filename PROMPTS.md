# Prompts

Every prompt used in the evaluation, verbatim. `{...}` are Python format fields.
Generation and judging both run through Ollama at `temperature=0`, `seed=42`.
Generator `mistral:7b-instruct` (7.2B, Q4_K_M); judge `llama3.1:8b` (8.0B, Q4_K_M).

## Generation — grounded arm

```text
You are a patient education assistant. Answer the patient's question using ONLY the information in the passages below.

Rules:
- Use only facts stated in the passages. Do not add anything from your own knowledge.
- After each fact, cite the passage it came from by its topic title in square brackets, like [Heart Failure].
{reading_level}- If the passages do not contain the information needed to answer, say exactly: \"I don't have reliable information to answer that. Please ask your health care provider.\"
- Do not give a diagnosis. Do not recommend changing any medicine or dose.

Passages:
{passages}

Patient's question: {question}

Answer:
```

## Generation — no-retrieval control

```text
You are a patient education assistant. Answer the patient's question.

Rules:
{reading_level}- Do not give a diagnosis. Do not recommend changing any medicine or dose.

Patient's question: {question}

Answer:
```

## Judge — faithfulness (per sentence)

```text
Here are passages from a medical reference, and one sentence from an answer that was written using them.

Passages:
{passages}

Sentence: \"{sentence}\"

Is the sentence supported by the passages? A sentence counts as supported if its factual claims appear in the passages, even if it is worded differently or uses simpler words. General advice to consult a doctor counts as supported.

Reply with exactly one word: YES or NO.
```

## Judge — answer relevancy

```text
A patient asked: \"{question}\"

They received this answer:
\"{answer}\"

On a scale of 1 to 5, how well does the answer address the patient's actual question? 1 = not at all, 3 = partially, 5 = fully addresses it.

Reply with only the number.
```

## Judge — context precision (per passage)

```text
A patient asked: "{question}"

Here is one passage that was retrieved to help answer it:
\"\"\"{chunk}\"\"\"

Does this passage contain information that helps answer the patient's question? Answer YES if it is useful, NO if it is off-topic or unhelpful.

Reply with exactly one word: YES or NO.
```

## Reading-level clause

Substituted into `{reading_level}` in both generation prompts. This clause is the
only difference between the standard and simplified conditions.

```python
READING_LEVELS = {
    'standard': '- Write in plain, everyday language a patient can understand.\n',

    'simplified': ('- Write for a reading age of 9 to 11 years, as recommended by NHS '
                   'guidance for patient information. Use short sentences. Use everyday '
                   'words instead of medical terms where you can, and explain any medical '
                   'term you cannot avoid.\n'),
}
```
