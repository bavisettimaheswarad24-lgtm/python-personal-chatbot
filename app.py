"""
MAHESH AI - Personal Python Question-Answering Chatbot
-------------------------------------------------------
High-performance Python Q&A assistant powered by an Inverted Index +
Sublinear TF-IDF Vector Space Model with Cosine Similarity.

Improvements:
  - Bigram/N-gram tokenization for context-aware phrase matching
  - Negation penalty system (without/not/no/never/avoid/except)
  - Multi-topic synthesis for cross-topic queries
  - Built-in exception type recognition
  - Expanded synonyms and knowledge base
"""

import json
import math
import re
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

KB_PATH = Path(__file__).with_name("knowledge_base.json")
CONFIDENCE_THRESHOLD = 0.18
SUGGESTION_THRESHOLD = 0.08
SECONDARY_THRESHOLD = 0.13   # second topic must be at least this to trigger synthesis
NEGATION_PENALTY = 0.15      # multiply IDF weight of negated tokens by this factor
NEGATION_WINDOW = 2          # number of tokens after a negation trigger to penalise

NEGATION_TRIGGERS = {"without", "not", "no", "never", "avoid", "except", "excluding", "skip", "instead"}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "do", "does", "did", "i", "me",
    "my", "you", "your", "we", "it", "its", "in", "on", "of", "to", "for",
    "and", "or", "how", "what", "why", "when", "which", "can", "could",
    "should", "would", "please", "tell", "explain", "about", "python", "use",
    "using", "way", "some", "there", "this", "that", "between", "get", "make",
}

# Expanded synonym table
SYNONYMS = {
    # collections
    "dictionary": "dict", "dictionaries": "dict", "dicts": "dict",
    "lists": "list", "arrays": "list", "array": "list",
    "strings": "string", "str": "string", "text": "string",
    "tuples": "tuple", "sets": "set",
    # errors
    "errors": "error", "exceptions": "exception", "catch": "except",
    "typeerror": "type_error", "valueerror": "value_error",
    "keyerror": "key_error", "indexerror": "index_error",
    "zerodivisionerror": "zero_division_error",
    "attributeerror": "attribute_error", "nameerror": "name_error",
    "runtimeerror": "runtime_error", "stopiteration": "stop_iteration",
    # functions / oop
    "functions": "function", "func": "function", "method": "function",
    "classes": "class", "objects": "object",
    # loops / iteration
    "loops": "loop", "looping": "loop",
    "iterating": "iterate", "iteration": "iterate", "iterable": "iterate",
    "recursion": "recursive", "recursive": "recursive",
    # files
    "files": "file", "reading": "read", "writing": "write",
    # modules
    "modules": "module", "packages": "package", "libraries": "library",
    # misc
    "installing": "install", "sorting": "sort", "decorators": "decorator",
    "generators": "generator", "variables": "variable", "types": "type",
    "venv": "virtual", "conversion": "convert", "casting": "convert",
    "cast": "convert", "lambdas": "lambda", "comprehensions": "comprehension",
    "arguments": "argument", "args": "argument", "parameters": "argument",
    "params": "argument", "parameter": "argument", "defaults": "default",
    "copies": "copy", "copying": "copy", "dunder": "magic",
    "threads": "thread", "threading": "thread", "multithreading": "thread",
    "coroutine": "async", "coroutines": "async", "asynchronous": "async",
    "concurrency": "async", "concurrent": "async",
    "gc": "garbage", "reference": "reference", "circular": "circular",
    "network": "socket", "networking": "socket", "sockets": "socket",
    "dataframe": "pandas", "dataframes": "pandas", "df": "pandas",
    "clean": "pep8", "style": "pep8", "lint": "pep8", "linting": "pep8",
    "functional": "functional", "immutably": "immutable",
    "map": "map", "filter": "filter", "reduce": "reduce",
    "walrus": "walrus", "assignment": "assign",
}

GREETINGS = {"hi", "hello", "hey", "hola", "yo"}
THANKS = {"thanks", "thank", "thx", "ty"}

# Recognised Python built-in exception names (kept as tokens, never stripped)
BUILTIN_EXCEPTIONS = {
    "typeerror", "valueerror", "keyerror", "indexerror", "zerodivisionerror",
    "attributeerror", "nameerror", "runtimeerror", "stopiteration",
    "overflowerror", "memoryerror", "recursionerror", "oserror",
    "filenotfounderror", "permissionerror", "importerror", "syntaxerror",
}


def _raw_tokens(text: str) -> list[str]:
    """Extract raw word tokens (including exception class names) from text."""
    words = re.findall(r"[a-z][a-z0-9_]*", text.lower())
    return words


def tokenize(text: str, include_bigrams: bool = True) -> list[str]:
    """
    Full tokenization pipeline:
      1. Extract raw words
      2. Apply synonym normalisation
      3. Remove stopwords (keep short special tokens and exception names)
      4. Optionally append bigrams of adjacent non-stopword tokens
         and negation bigrams (e.g. without_loop).
    """
    raw = _raw_tokens(text)
    normalized = [SYNONYMS.get(w, w) for w in raw]
    unigrams = [
        w for w in normalized
        if (w not in STOPWORDS and len(w) > 1) or w in {"=", "@"} or w in BUILTIN_EXCEPTIONS
    ]
    if not include_bigrams:
        return unigrams

    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]

    # Also capture negation phrase bigrams (e.g., without_loop, avoid_loop)
    for i in range(len(raw) - 1):
        w1, w2 = SYNONYMS.get(raw[i], raw[i]), SYNONYMS.get(raw[i + 1], raw[i + 1])
        if (w1 in NEGATION_TRIGGERS or w2 in NEGATION_TRIGGERS) and len(w1) > 1 and len(w2) > 1:
            bg = f"{w1}_{w2}"
            if bg not in bigrams:
                bigrams.append(bg)

    return unigrams + bigrams


def _build_negation_mask(text: str) -> set[str]:
    """
    Return a set of token strings that appear in the scope following a negation
    trigger word in the raw query, so they can be penalised during scoring.
    """
    raw = _raw_tokens(text)
    penalised = set()
    for i, w in enumerate(raw):
        if w in NEGATION_TRIGGERS:
            count = 0
            for j in range(i + 1, min(i + 10, len(raw))):
                word = raw[j]
                norm = SYNONYMS.get(word, word)
                if norm not in STOPWORDS and len(norm) > 1:
                    penalised.add(norm)
                    count += 1
                    if count >= 4:
                        break
    return penalised


class OptimizedPythonBot:
    def __init__(self, kb_path: Path = KB_PATH):
        with open(kb_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        self._build_inverted_index()

    def _build_inverted_index(self) -> None:
        docs = []
        for entry in self.kb:
            text = " ".join(entry["questions"]) + " " + entry["topic"]
            text += (" " + " ".join(entry["keywords"]) + " " + entry["topic"]) * 3
            text += " " + entry.get("answer", "")
            docs.append(Counter(tokenize(text)))

        n = len(docs)
        df = Counter(t for doc in docs for t in doc)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vocab = list(self.idf)

        # Precompute L2-normalised document vectors
        self.doc_vectors = []
        for doc in docs:
            vec = {t: (1.0 + math.log(c)) * self.idf[t] for t, c in doc.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors.append({t: v / norm for t, v in vec.items()})

        # Build Inverted Posting List for fast search
        self.inverted_index = defaultdict(list)
        for doc_id, vec in enumerate(self.doc_vectors):
            for term, weight in vec.items():
                self.inverted_index[term].append((doc_id, weight))

    def _correct_spelling(self, tokens: list[str]) -> list[str]:
        fixed = []
        for t in tokens:
            if t in self.idf:
                fixed.append(t)
            elif "_" in t:
                # bigrams: try to correct each part independently
                parts = t.split("_", 1)
                corrected = []
                for p in parts:
                    m = get_close_matches(p, self.vocab, n=1, cutoff=0.8)
                    corrected.append(m[0] if m else p)
                fixed.append("_".join(corrected))
            else:
                m = get_close_matches(t, self.vocab, n=1, cutoff=0.8)
                fixed.append(m[0] if m else t)
        return fixed

    def query(self, question: str) -> list[tuple[float, dict]]:
        tokens = self._correct_spelling(tokenize(question))
        negated = _build_negation_mask(question)

        if not tokens and "python" in question.lower():
            tokens = ["introduction", "language"]
        if not tokens:
            return []

        counts = Counter(tokens)
        qvec: dict[str, float] = {}
        for t, c in counts.items():
            idf_val = self.idf.get(t, 0)
            if idf_val == 0:
                continue
            weight = (1.0 + math.log(c)) * idf_val
            # Apply negation penalty: if this token or its constituent sub-words were flagged
            # as negated (and it is not an explicit negation phrase like without_loop),
            # down-weight it so matching documents that contain the excluded concept score lower.
            is_neg = False
            if not any(t.startswith(neg + "_") for neg in NEGATION_TRIGGERS):
                if t in negated or any(p in negated for p in t.split("_")):
                    is_neg = True
            if is_neg:
                weight *= NEGATION_PENALTY
            qvec[t] = weight

        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        qvec = {t: v / qnorm for t, v in qvec.items()}

        scores: dict[int, float] = defaultdict(float)
        for term, qweight in qvec.items():
            for doc_id, doc_weight in self.inverted_index.get(term, []):
                scores[doc_id] += qweight * doc_weight

        ranked = [(score, self.kb[doc_id]) for doc_id, score in scores.items()]
        ranked.sort(key=lambda s: s[0], reverse=True)
        return ranked

    def answer(self, question: str) -> str:
        words = set(re.findall(r"[a-z]+", question.lower()))
        if words and words <= GREETINGS | {"there", "bot", "mahesh"}:
            return (
                "Hello! I am **MAHESH AI**, your personal Python Q&A assistant. "
                "Ask me any Python question (e.g., *'What is the difference between "
                "list and tuple?'*, *'How do decorators work?'*, *'Explain GIL'*)."
            )
        if words & THANKS and len(words) <= 4:
            return "You're welcome! Feel free to ask another Python question."

        q_lower = question.strip().lower()
        if q_lower in {"topics", "help"}:
            return (
                "### Available Python Topics (" + str(len(self.kb)) + ")\n\n"
                + "\n".join(f"* {e['topic']}" for e in self.kb)
            )

        ranked = self.query(question)
        if not ranked:
            return (
                "Please type a Python question, for example: "
                "*'How do I reverse a string?'* or *'What does yield do?'*"
            )

        best_score, best = ranked[0]

        # --- Multi-topic synthesis ---
        # Collect all results above CONFIDENCE_THRESHOLD
        confident = [(s, e) for s, e in ranked if s >= CONFIDENCE_THRESHOLD]
        secondary = [
            (s, e) for s, e in ranked[1:]
            if s >= SECONDARY_THRESHOLD and e["topic"] != best["topic"]
        ]

        if best_score >= CONFIDENCE_THRESHOLD:
            if len(confident) >= 2 and secondary:
                # Build a composite multi-topic response
                second_score, second = secondary[0]
                reply = (
                    f"### [{best['topic']}]\n\n{best['answer']}\n\n"
                    f"---\n\n"
                    f"### Also relevant → [{second['topic']}]\n\n{second['answer']}"
                )
                # Any further related topics
                extra = [e["topic"] for s, e in ranked[2:5] if s >= SUGGESTION_THRESHOLD]
                extra = [t for t in extra if t != best["topic"] and t != second["topic"]]
                if extra:
                    reply += "\n\n**See also:** " + ", ".join(f"`{r}`" for r in extra)
            else:
                reply = f"### [{best['topic']}]\n\n{best['answer']}"
                related = [e["topic"] for s, e in ranked[1:4] if s >= SUGGESTION_THRESHOLD]
                if related:
                    reply += "\n\n**Related topics:** " + ", ".join(f"`{r}`" for r in related)
            return reply

        suggestions = [e["topic"] for s, e in ranked[:3] if s >= SUGGESTION_THRESHOLD]
        if suggestions:
            return (
                "I'm not completely sure. Did you mean one of these topics?\n\n* "
                + "\n* ".join(f"`{s}`" for s in suggestions)
            )
        return (
            "Sorry, I don't have an answer for that yet. "
            "Try rephrasing your Python question or ask about loops, dicts, functions, or OOP."
        )


bot = OptimizedPythonBot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty question."}), 400
    reply = bot.answer(message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
