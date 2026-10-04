"""
Patient education RAG demo, with optional condition profile.

Answers health questions using only retrieved MedlinePlus passages, cites every
claim back to a topic, and refuses when the corpus doesn't cover the question.

Optionally the patient can say what condition they have. That does two things:
chunks from their condition's topic get a small ranking bonus during retrieval,
and the prompt is told the condition so the answer can be framed for them. It
never relaxes the grounding rules. Leave it blank and the app behaves exactly
as app.py does.

This is a separate entry point. It does not import from or modify app.py, and
it opens chroma_db read-only, so the plain demo is unaffected.

Run from the same folder as chroma_db/:
    streamlit run app_personalised.py

To run it alongside the plain demo, give it its own port:
    streamlit run app_personalised.py --server.port 8502

Needs ollama running (ollama serve) with mistral:7b-instruct pulled.
"""

import re

import chromadb
import ollama
import streamlit as st
import textstat
from sentence_transformers import SentenceTransformer

# ---- same config as the evaluation notebooks, so the demo shows what I measured ----
GENERATOR_MODEL = "mistral:7b-instruct"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DB_DIR = "chroma_db"
COLLECTION = "medlineplus_english"
TOP_K = 5
MAX_PER_TOPIC = 2
TEMPERATURE = 0
SEED = 42

# personalisation - fixed in advance, never tuned against results
CONDITION_BOOST = 0.1
CONDITION_MIN_SIMILARITY = 0.55

# below this the corpus probably doesn't cover the question. eyeballed from the
# evaluation runs where similarities sat around 0.6-0.87 - a rule of thumb, not
# a calibrated confidence score, and labelled as such in the UI
LOW_CONFIDENCE = 0.65

REFUSAL_MARKER = "i don't have reliable information"

READING_LEVELS = {
    "standard": "- Write in plain, everyday language a patient can understand.\n",

    "simplified": ("- Write for a reading age of 9 to 11 years, as recommended by NHS "
                   "guidance for patient information. Use short sentences. Use everyday "
                   "words instead of medical terms where you can, and explain any medical "
                   "term you cannot avoid.\n"),
}

# what the sidebar shows, mapped to the keys above.
# DEFAULT_LEVEL is what the box opens on. Keep it "standard" - that is the
# arm the headline held-out numbers were measured under.
DEFAULT_LEVEL = "standard"

LEVEL_LABELS = {
    "Standard — plain language": "standard",
    "Simplified — NHS reading age 9–11": "simplified",
}

# derived, so the default can never drift if the labels are reordered
DEFAULT_LEVEL_INDEX = list(LEVEL_LABELS.values()).index(DEFAULT_LEVEL)

RAG_PROMPT = """You are a patient education assistant. Answer the patient's question using ONLY the information in the passages below.

Rules:
- Use only facts stated in the passages. Do not add anything from your own knowledge.
- After each fact, cite the passage it came from by its topic title in square brackets, like [Heart Failure].
{reading_level}- If the passages do not contain the information needed to answer, say exactly: "I don't have reliable information to answer that. Please ask your health care provider."
- Do not give a diagnosis. Do not recommend changing any medicine or dose.

Passages:
{passages}

Patient's question: {question}

Answer:"""

# identical rules. the condition frames the answer, it does not license anything
# that isn't in the passages.
PERSONALISED_PROMPT = """You are a patient education assistant. Answer the patient's question using ONLY the information in the passages below.

The patient has told us their condition or recent procedure is: {condition}. Use this to frame the answer for them, but it does not change what you may say.

Rules:
- Use only facts stated in the passages. Do not add anything from your own knowledge, including anything you know about the condition above.
- After each fact, cite the passage it came from by its topic title in square brackets, like [Heart Failure].
{reading_level}- If the passages do not contain the information needed to answer, say exactly: "I don't have reliable information to answer that. Please ask your health care provider."
- Do not give a diagnosis. Do not recommend changing any medicine or dose.

Passages:
{passages}

Patient's question: {question}

Answer:"""

BASELINE_PROMPT = """You are a patient education assistant. Answer the patient's question.

Rules:
{reading_level}- Do not give a diagnosis. Do not recommend changing any medicine or dose.

Patient's question: {question}

Answer:"""

EXAMPLES = [
    "What should I do after being discharged with heart failure?",
    "blood thinner side effects",
    "What are the symptoms of Shingles?",
    "Can you renew my car insurance for me?",
]

CITE_RE = re.compile(r"\[([^\]]+)\]")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the index and embedding model...")
def load():
    embedder = SentenceTransformer(EMBED_MODEL)
    coll = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return embedder, coll


def embed_query(embedder, text):
    return embedder.encode("Represent this sentence for searching relevant passages: " + text)


def retrieve(embedder, coll, query, k=TOP_K, max_per_topic=MAX_PER_TOPIC):
    """same retriever as the notebooks - over-fetch, then cap chunks per topic so
    one topic can't take every slot. unchanged from app.py apart from two extra
    keys on each hit so the display code can treat both retrievers alike."""
    res = coll.query(query_embeddings=[embed_query(embedder, query).tolist()], n_results=k * 10)

    hits, spill, per_topic = [], [], {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hit = {
            "text": doc,
            "title": meta["title"],
            "url": meta["url"],
            "group": meta["group"],
            "similarity": round(1 - dist, 3),
            "raw_similarity": round(1 - dist, 3),
            "boosted": False,
        }
        seen = per_topic.get(meta["topic_id"], 0)
        if seen >= max_per_topic:
            spill.append(hit)
            continue
        per_topic[meta["topic_id"]] = seen + 1
        hits.append(hit)
        if len(hits) == k:
            break
    if len(hits) < k:
        hits.extend(spill[: k - len(hits)])
    return hits


def match_condition_to_topic(embedder, coll, condition_text,
                             min_similarity=CONDITION_MIN_SIMILARITY):
    """Free text ('had my knee done') -> the nearest corpus topic title.

    Queries the existing collection rather than building a second index. Every
    chunk carries its title and synonyms on the front, so a condition name
    matches its own topic strongly. Returns the match and its similarity so the
    interface can show the user what the system thinks they have.
    """
    if not condition_text or not condition_text.strip():
        return None, 0.0
    res = coll.query(query_embeddings=[embed_query(embedder, condition_text).tolist()],
                     n_results=5)
    metas, dists = res["metadatas"][0], res["distances"][0]
    if not metas:
        return None, 0.0
    title, sim = metas[0]["title"], round(1 - dists[0], 3)
    return (title, sim) if sim >= min_similarity else (None, sim)


def retrieve_personalised(embedder, coll, query, condition_topic=None,
                          k=TOP_K, max_per_topic=MAX_PER_TOPIC, boost=CONDITION_BOOST):
    """Retrieve as normal, then give chunks from the patient's condition topic a
    fixed bonus and re-rank. A re-rank, not a filter: a chunk from another topic
    still wins if it is a clearly better match, which matters because the
    question is often not about the condition itself.

    With condition_topic=None this delegates to retrieve() and returns exactly
    what the plain demo would return.
    """
    if not condition_topic:
        return retrieve(embedder, coll, query, k, max_per_topic)

    res = coll.query(query_embeddings=[embed_query(embedder, query).tolist()], n_results=k * 10)

    cands = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        raw = round(1 - dist, 3)
        boosted = meta["title"] == condition_topic
        cands.append({
            "text": doc, "title": meta["title"], "url": meta["url"],
            "group": meta["group"], "topic_id": meta["topic_id"],
            "similarity": round(raw + boost, 3) if boosted else raw,
            "raw_similarity": raw, "boosted": boosted,
        })

    cands.sort(key=lambda h: -h["similarity"])

    hits, spill, per_topic = [], [], {}
    for hit in cands:
        seen = per_topic.get(hit["topic_id"], 0)
        if seen >= max_per_topic:
            spill.append(hit)
            continue
        per_topic[hit["topic_id"]] = seen + 1
        hits.append(hit)
        if len(hits) == k:
            break
    if len(hits) < k:
        hits.extend(spill[: k - len(hits)])
    return hits


def format_passages(hits):
    return "\n\n---\n\n".join(f"Topic: {h['title']}\n{h['text']}" for h in hits)


def stream_answer(prompt):
    """generate token by token so the demo doesn't sit blank for 20 seconds"""
    for chunk in ollama.chat(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"temperature": TEMPERATURE, "seed": SEED},
    ):
        yield chunk["message"]["content"]


def strip_citations(text):
    return re.sub(r"\s{2,}", " ", CITE_RE.sub("", text)).strip()


def answer_metrics(answer, hits):
    clean = strip_citations(answer)
    words = len(clean.split())
    cited = [c.strip() for c in CITE_RE.findall(answer)]
    retrieved = {h["title"].lower() for h in hits}
    valid = sum(1 for c in cited if c.lower() in retrieved)
    return {
        "fkgl": round(textstat.flesch_kincaid_grade(clean), 1) if words >= 10 else None,
        "fre": round(textstat.flesch_reading_ease(clean), 1) if words >= 10 else None,
        "words": words,
        "citations": len(cited),
        "citations_valid": valid,
    }


def show_metrics(m):
    c1, c2, c3, c4 = st.columns(4)
    if m["fkgl"] is not None:
        # NHS guidance is a reading age of 9-11, which is roughly FKGL 4-6.
        # 6 is the lenient end of that band.
        target = "at NHS target" if m["fkgl"] <= 6 else "above NHS target"
        c1.metric("Reading grade", m["fkgl"], target, delta_color="off")
        c2.metric("Reading ease", m["fre"])
    else:
        c1.metric("Reading grade", "—")
        c2.metric("Reading ease", "—")
    c3.metric("Words", m["words"])
    if m["citations"]:
        c4.metric("Citations", f"{m['citations_valid']}/{m['citations']}",
                  "traced to source", delta_color="off")
    else:
        c4.metric("Citations", "0")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Patient Education RAG (personalised)", layout="wide")
st.title("AI-Supported Patient Education")
st.caption(
    "Answers are generated only from MedlinePlus passages retrieved for your question. "
    "Every claim is cited back to its source topic."
)

embedder, coll = load()

with st.sidebar:
    st.subheader("Settings")
    level_label = st.selectbox("Reading level", list(LEVEL_LABELS), index=DEFAULT_LEVEL_INDEX)
    level_key = LEVEL_LABELS[level_label]
    show_baseline = st.toggle(
        "Compare against no-RAG baseline",
        help="Same model answering from its own memory, with no retrieved passages. "
        "This is the control arm from the evaluation.",
    )
    st.divider()
    st.subheader("Configuration")
    st.caption(
        f"**Generator** {GENERATOR_MODEL}  \n"
        f"**Embeddings** {EMBED_MODEL}  \n"
        f"**Corpus** {coll.count():,} chunks / 1,016 English topics  \n"
        f"**Retrieval** top-{TOP_K}, max {MAX_PER_TOPIC} per topic  \n"
        f"**Condition bonus** +{CONDITION_BOOST} when a profile is given  \n"
        f"**Temperature** {TEMPERATURE} (seed {SEED})"
    )
    st.divider()
    st.caption(
        "Demo only. Not medical advice, and not a substitute for a "
        "health care provider."
    )

st.write("**Try one of these:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex if len(ex) < 34 else ex[:31] + "...", use_container_width=True, help=ex):
        st.session_state.question = ex

# ---- optional condition profile -------------------------------------------
condition = st.text_input(
    "Your condition or recent procedure (optional)",
    key="condition",
    placeholder="e.g. knee replacement, COPD, heart failure",
    help="Optional. If you tell us this, answers are steered toward material about "
         "your condition. Leave it blank and the system behaves exactly as it would "
         "otherwise.",
)

matched_topic, match_sim = (None, 0.0)
if condition:
    matched_topic, match_sim = match_condition_to_topic(embedder, coll, condition)
    if matched_topic:
        st.caption(
            f"Matched to the MedlinePlus topic **{matched_topic}** (similarity "
            f"{match_sim}). If that is not right, try different wording."
        )
    else:
        st.caption(
            f"No corpus topic matched closely enough (best similarity {match_sim}). "
            "Answering without a condition profile."
        )
st.caption("Your condition is used for this question only. It is never stored.")

question = st.text_input(
    "Ask a health question",
    key="question",
    placeholder="e.g. How do I care for a wound at home?",
)

if question:
    hits = retrieve_personalised(embedder, coll, question, matched_topic)

    # the boost inflates the score, so judge coverage on the best unboosted value
    top_sim = max((h["raw_similarity"] for h in hits), default=0)

    if top_sim < LOW_CONFIDENCE:
        st.warning(
            f"Closest match scored {top_sim:.2f}. The corpus may not cover this question well — "
            "the system should decline rather than guess. (Similarity threshold is a rule of "
            "thumb, not a calibrated confidence score.)"
        )

    if matched_topic:
        n_boost = sum(1 for h in hits if h.get("boosted"))
        st.caption(
            f"{n_boost} of {len(hits)} passages came from your condition topic "
            f"({matched_topic})."
        )

    level = READING_LEVELS[level_key]

    if matched_topic:
        rag_prompt = PERSONALISED_PROMPT.format(
            passages=format_passages(hits), question=question,
            reading_level=level, condition=matched_topic,
        )
    else:
        rag_prompt = RAG_PROMPT.format(
            passages=format_passages(hits), question=question, reading_level=level,
        )

    if show_baseline:
        left, right = st.columns(2)
        left.subheader("With retrieval")
        left.caption("Grounded in MedlinePlus, every claim cited")
        with left:
            rag_answer = st.write_stream(stream_answer(rag_prompt))
            show_metrics(answer_metrics(rag_answer, hits))

        right.subheader("Without retrieval")
        right.caption("Same model, answering from memory — no sources to check")
        with right:
            base_answer = st.write_stream(
                stream_answer(BASELINE_PROMPT.format(question=question, reading_level=level))
            )
            show_metrics(answer_metrics(base_answer, hits))
            st.info(
                "Nothing here can be traced to a source. On the held-out evaluation this arm "
                "scored 0.83 faithfulness against 1.00 with retrieval."
            )
    else:
        rag_answer = st.write_stream(stream_answer(rag_prompt))
        if REFUSAL_MARKER in rag_answer.lower():
            st.success(
                "The system declined rather than guessing — this is the intended behaviour "
                "when the corpus doesn't cover a question."
            )
        show_metrics(answer_metrics(rag_answer, hits))

    with st.expander(f"Sources used ({len(hits)} passages retrieved)"):
        for h in hits:
            tag = "  ·  boosted by your condition" if h.get("boosted") else ""
            sim = h["raw_similarity"] if h.get("boosted") else h["similarity"]
            st.markdown(
                f"**[{h['title']}]({h['url']})** · {h['group']} · similarity {sim}{tag}"
            )
            st.caption(h["text"][:320] + ("..." if len(h["text"]) > 320 else ""))
            st.divider()
