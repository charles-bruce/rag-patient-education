"""
Patient education RAG demo.

Answers health questions using only retrieved MedlinePlus passages, cites every
claim back to a topic, and refuses when the corpus doesn't cover the question.

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

# below this the corpus probably doesn't cover the question. eyeballed from the
# evaluation runs where similarities sat around 0.6-0.87 - a rule of thumb, not
# a calibrated confidence score, and labelled as such in the UI
LOW_CONFIDENCE = 0.65

REFUSAL_MARKER = "i don't have reliable information"

READING_LEVELS = {
    'standard': '- Write in plain, everyday language a patient can understand.\n',

    'simplified': ('- Write for a reading age of 9 to 11 years, as recommended by NHS '
                   'guidance for patient information. Use short sentences. Use everyday '
                   'words instead of medical terms where you can, and explain any medical '
                   'term you cannot avoid.\n'),
}

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


def retrieve(embedder, coll, query, k=TOP_K, max_per_topic=MAX_PER_TOPIC):
    """same retriever as the notebooks - over-fetch, then cap chunks per topic so
    one topic can't take every slot"""
    q_emb = embedder.encode("Represent this sentence for searching relevant passages: " + query)
    res = coll.query(query_embeddings=[q_emb.tolist()], n_results=k * 10)

    hits, spill, per_topic = [], [], {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hit = {
            "text": doc,
            "title": meta["title"],
            "url": meta["url"],
            "group": meta["group"],
            "similarity": round(1 - dist, 3),
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
        # NHS guidance is a reading age of 9-11, roughly FKGL 4-6. Six is the
        # lenient end of that band.
        target = "at NHS target" if m["fkgl"] <= 6 else "above NHS target"
        c1.metric("Reading grade", m["fkgl"], target, delta_color="off")
        c2.metric("Reading ease", m["fre"])
    else:
        c1.metric("Reading grade", "—")
        c2.metric("Reading ease", "—")
    c3.metric("Words", m["words"])
    if m["citations"]:
        c4.metric("Citations", f"{m['citations_valid']}/{m['citations']}", "traced to source", delta_color="off")
    else:
        c4.metric("Citations", "0")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Patient Education RAG", layout="wide")
st.title("AI-Supported Patient Education")
st.caption(
    "Answers are generated only from MedlinePlus passages retrieved for your question. "
    "Every claim is cited back to its source topic."
)

embedder, coll = load()

with st.sidebar:
    st.subheader("Settings")
    level_name = st.selectbox("Reading level", list(READING_LEVELS), index=0)
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

question = st.text_input(
    "Ask a health question",
    key="question",
    placeholder="e.g. How do I care for a wound at home?",
)

if question:
    hits = retrieve(embedder, coll, question)
    top_sim = hits[0]["similarity"] if hits else 0

    if top_sim < LOW_CONFIDENCE:
        st.warning(
            f"Closest match scored {top_sim:.2f}. The corpus may not cover this question well — "
            "the system should decline rather than guess. (Similarity threshold is a rule of "
            "thumb, not a calibrated confidence score.)"
        )

    level = READING_LEVELS[level_name]
    rag_prompt = RAG_PROMPT.format(
        passages=format_passages(hits), question=question, reading_level=level
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
            st.markdown(
                f"**[{h['title']}]({h['url']})** · {h['group']} · similarity {h['similarity']}"
            )
            st.caption(h["text"][:320] + ("..." if len(h["text"]) > 320 else ""))
            st.divider()
