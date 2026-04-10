import json
import math
import os
import platform
import re
import tempfile
import textwrap
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st


STOPWORDS_IT = {
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli", "all", "anche", "ancora",
    "avere", "aveva", "avevano", "basta", "bene", "che", "chi", "ci", "cioè", "come",
    "con", "contro", "cosa", "cui", "da", "dagli", "dai", "dal", "dalla", "dalle",
    "dallo", "de", "dei", "del", "della", "delle", "dello", "di", "dopo", "dove",
    "due", "e", "ed", "era", "erano", "essere", "fa", "faccio", "fare", "fatto",
    "fra", "gli", "ha", "hai", "hanno", "ho", "i", "il", "in", "io", "la", "le",
    "lei", "li", "lo", "loro", "lui", "ma", "mi", "mia", "mie", "miei", "mio",
    "molto", "nei", "nel", "nella", "nelle", "nello", "noi", "non", "nostro", "o",
    "oggi", "ogni", "per", "perché", "più", "poi", "poco", "può", "quale", "quali",
    "quello", "questa", "queste", "questi", "questo", "qui", "quindi", "sarà", "se",
    "sei", "senza", "si", "sia", "siamo", "sono", "sopra", "sotto", "sua", "sue",
    "suoi", "sul", "sulla", "sulle", "sullo", "su", "tra", "tre", "tu", "tua", "tuo",
    "un", "una", "uno", "va", "voi", "vostra", "vostro", "è", "eh", "allora",
    "diciamo", "praticamente", "tipo", "ok", "cioe", "insomma", "dunque",
}

FILLERS = [
    r"\behm+\b",
    r"\bem+\b",
    r"\bmh+\b",
    r"\bmmm+\b",
    r"\bcioè\b",
    r"\bdiciamo\b",
    r"\bpraticamente\b",
    r"\bok\b",
    r"\ballora\b",
    r"\binsomma\b",
]

APPLE_MLX_MODELS = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3",
    "medium",
    "small",
    "base",
    "tiny",
]

FASTER_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "mlx-community/whisper-large-v3-turbo",
]


def is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9àèéìòù_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "lezione"


def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_transcript_text(text: str) -> str:
    t = text or ""
    for patt in FILLERS:
        t = re.sub(patt, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])([A-Za-zÀ-ÿ])", r"\1 \2", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \n\t")
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [normalize_spaces(p) for p in parts if normalize_spaces(p)]


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']+", text.lower())
    return [w for w in words if w not in STOPWORDS_IT and len(w) > 2]


def extract_keywords(text: str, top_n: int = 12) -> List[str]:
    counts = Counter(tokenize(text))
    return [w for w, _ in counts.most_common(top_n)]


def build_extract_summary(text: str, max_sentences: int = 8) -> List[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    freq = Counter(tokenize(text))
    scored: List[Tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        words = tokenize(sent)
        if not words:
            continue
        score = sum(freq[w] for w in words) / math.sqrt(max(1, len(words)))
        scored.append((score, idx, sent))
    top = sorted(scored, reverse=True)[:max_sentences]
    chosen = sorted(top, key=lambda x: x[1])
    return [c[2] for c in chosen]


def group_segments_to_paragraphs(segments: List[Dict], max_chars: int = 1200) -> List[str]:
    paragraphs: List[str] = []
    current: List[str] = []
    current_len = 0

    for seg in segments:
        text = normalize_spaces(seg.get("text", ""))
        if not text:
            continue
        if current and current_len + len(text) + 1 > max_chars:
            paragraphs.append(clean_transcript_text(" ".join(current)))
            current = []
            current_len = 0
        current.append(text)
        current_len += len(text) + 1

    if current:
        paragraphs.append(clean_transcript_text(" ".join(current)))

    return [p for p in paragraphs if p]


def normalize_faster_model_name(model_name: str) -> str:
    if model_name in FASTER_WHISPER_MODELS:
        return model_name
    if "large-v3" in model_name:
        return "large-v3"
    if model_name in {"tiny", "base", "small", "medium"}:
        return model_name
    return "medium"


def normalize_mlx_model_name(model_name: str) -> str:
    if model_name in APPLE_MLX_MODELS:
        return model_name
    if model_name == "large-v3":
        return "mlx-community/whisper-large-v3"
    if model_name in {"tiny", "base", "small", "medium"}:
        return model_name
    return "mlx-community/whisper-large-v3-turbo"


def choose_default_model() -> str:
    return "mlx-community/whisper-large-v3-turbo" if is_apple_silicon_mac() else "medium"


# -----------------------------
# Whisper transcription
# -----------------------------
def transcribe_file_faster_whisper(
    file_path: str,
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    vad_filter: bool,
) -> Tuple[List[Dict], Dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        file_path,
        language=None if language == "auto" else language,
        vad_filter=vad_filter,
        beam_size=5,
        word_timestamps=False,
    )

    segments = []
    for s in segments_iter:
        segments.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
            }
        )

    meta = {
        "backend": "faster-whisper",
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
    }
    return segments, meta



def transcribe_file_mlx(
    file_path: str,
    model_size: str,
    language: Optional[str],
) -> Tuple[List[Dict], Dict]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper non è installato. Su Apple Silicon puoi fare: pip install mlx-whisper"
        ) from exc

    result = mlx_whisper.transcribe(
        file_path,
        path_or_hf_repo=model_size,
        language=None if language == "auto" else language,
        word_timestamps=False,
        verbose=False,
    )

    raw_segments = result.get("segments", []) or []
    segments = []
    for s in raw_segments:
        segments.append(
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip(),
            }
        )

    meta = {
        "backend": "mlx-whisper",
        "language": result.get("language"),
        "duration": None,
        "model": model_size,
        "device": "apple-silicon",
        "compute_type": "mlx",
    }
    return segments, meta



def transcribe_file(
    file_path: str,
    backend: str,
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    vad_filter: bool,
) -> Tuple[List[Dict], Dict]:
    if backend == "mlx-whisper":
        return transcribe_file_mlx(file_path, normalize_mlx_model_name(model_size), language)

    if backend == "faster-whisper":
        return transcribe_file_faster_whisper(
            file_path=file_path,
            model_size=normalize_faster_model_name(model_size),
            language=language,
            device=device,
            compute_type=compute_type,
            vad_filter=vad_filter,
        )

    if backend == "auto":
        if is_apple_silicon_mac():
            try:
                return transcribe_file_mlx(file_path, normalize_mlx_model_name(model_size), language)
            except Exception:
                return transcribe_file_faster_whisper(
                    file_path=file_path,
                    model_size=normalize_faster_model_name(model_size),
                    language=language,
                    device="cpu",
                    compute_type="int8",
                    vad_filter=vad_filter,
                )
        return transcribe_file_faster_whisper(
            file_path=file_path,
            model_size=normalize_faster_model_name(model_size),
            language=language,
            device=device,
            compute_type=compute_type,
            vad_filter=vad_filter,
        )

    raise ValueError(f"Backend non supportato: {backend}")


# -----------------------------
# Optional Ollama integration
# -----------------------------
def ollama_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False



def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []



def strip_ollama_wrappers(raw: str) -> str:
    cleaned = (raw or "").replace("\ufeff", "").replace("\x00", "").strip()
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()



def extract_first_json_block(raw: str) -> Optional[str]:
    if not raw:
        return None

    start_positions = [idx for idx in [raw.find("{"), raw.find("[")] if idx != -1]
    if not start_positions:
        return None

    start = min(start_positions)
    opening = raw[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    return None



def sanitize_json_candidate(raw: str) -> str:
    cleaned = strip_ollama_wrappers(raw)
    translation_table = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }
    )
    cleaned = cleaned.translate(translation_table)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned.strip()



def parse_json_from_text(raw: str) -> Dict:
    cleaned = sanitize_json_candidate(raw)
    if not cleaned:
        raise ValueError("Ollama ha restituito una risposta vuota.")

    candidates = []
    for candidate in [cleaned, extract_first_json_block(cleaned)]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if not isinstance(parsed, dict):
                raise ValueError("Il JSON restituito non è un oggetto.")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    if last_error:
        raise ValueError(f"Risposta Ollama non parseabile come JSON: {last_error}") from last_error
    raise ValueError("Risposta Ollama non parseabile come JSON.")



def build_ollama_prompt(title: str, transcript: str, summary_mode: str) -> str:
    summary_specs = {
        "breve": "4-6 punti di sintesi",
        "standard": "6-8 punti di sintesi",
        "dettagliata": "10-12 punti di sintesi",
    }
    summary_instruction = summary_specs.get(summary_mode, summary_specs["standard"])

    return textwrap.dedent(
        f"""
        Sei un assistente universitario. Analizza la trascrizione di una lezione e restituisci SOLO JSON valido.

        Titolo lezione: {title}

        Regole:
        - Non inventare informazioni non presenti nella trascrizione.
        - Scrivi in italiano chiaro e ordinato.
        - Mantieni i concetti tecnici importanti.
        - Se un termine è ambiguo, dillo senza inventare.
        - Non aggiungere testo fuori dal JSON.

        JSON richiesto:
        {{
          "summary": ["..."],
          "key_concepts": [
            {{"term": "...", "explanation": "..."}}
          ],
          "chapters": [
            {{"title": "...", "content": "..."}}
          ],
          "glossary": [
            {{"term": "...", "definition": "..."}}
          ],
          "review_questions": ["..."]
        }}

        Vincoli:
        - summary: {summary_instruction}
        - key_concepts: 6-10 elementi
        - chapters: 3-8 sezioni, con contenuto discorsivo utile per studiare
        - glossary: 5-10 termini importanti
        - review_questions: 6-10 domande di ripasso

        Trascrizione:
        {transcript}
        """
    ).strip()



def call_ollama_study_pack(
    model: str,
    title: str,
    transcript: str,
    base_url: str = "http://localhost:11434",
    summary_mode: str = "standard",
    max_chars: int = 90000,
) -> Tuple[Dict, bool]:
    clipped = transcript[:max_chars]
    prompt = build_ollama_prompt(title=title, transcript=clipped, summary_mode=summary_mode)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    response = data.get("response", "")
    try:
        parsed = parse_json_from_text(response)
        return parsed, len(transcript) > max_chars
    except Exception as exc:
        preview = strip_ollama_wrappers(response)[:500]
        raise ValueError(
            "Ollama ha risposto, ma non con JSON valido. "
            f"Anteprima risposta: {preview or '[vuota]'}"
        ) from exc


def build_ollama_notes_prompt(title: str, study_pack: Dict, cleaned_text: str) -> str:
    return textwrap.dedent(
        f"""
        Sei un tutor universitario. Riscrivi gli appunti in formato Markdown, chiaro e pronto per studiare.
        Restituisci SOLO testo markdown.

        Titolo: {title}

        Requisiti:
        - Struttura con sezioni e sottosezioni.
        - Evidenzia definizioni, formule e punti critici.
        - Aggiungi mini-riassunto finale (5-8 bullet).
        - Non inventare concetti non presenti.
        - Linguaggio semplice e preciso.

        Pacchetto studio (JSON):
        {json.dumps(study_pack, ensure_ascii=False)}

        Testo trascritto pulito:
        {cleaned_text[:80000]}
        """
    ).strip()


def call_ollama_enhance_notes(
    model: str,
    title: str,
    study_pack: Dict,
    cleaned_text: str,
    base_url: str = "http://localhost:11434",
) -> str:
    payload = {
        "model": model,
        "prompt": build_ollama_notes_prompt(title, study_pack, cleaned_text),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=600)
    response.raise_for_status()
    raw = response.json().get("response", "")
    return strip_ollama_wrappers(raw)


# -----------------------------
# Study pack generation
# -----------------------------
def find_evidence_sentence(term: str, sentences: List[str]) -> str:
    pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    for sent in sentences:
        if pattern.search(sent):
            return sent
    return ""



def build_chapter_title(text: str, index: int) -> str:
    keywords = extract_keywords(text, top_n=4)
    if keywords:
        pretty = ", ".join(k.capitalize() for k in keywords[:3])
        return f"Sezione {index}: {pretty}"
    return f"Sezione {index}"



def build_fallback_chapters(paragraphs: List[str]) -> List[Dict[str, str]]:
    if not paragraphs:
        return []

    total_chars = sum(len(p) for p in paragraphs)
    target_chapters = max(3, min(6, total_chars // 3500 if total_chars > 0 else 3))
    target_chars = max(1800, total_chars // max(1, target_chapters))

    chapters: List[Dict[str, str]] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > target_chars:
            content = "\n\n".join(current)
            chapters.append({"title": build_chapter_title(content, len(chapters) + 1), "content": content})
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 2

    if current:
        content = "\n\n".join(current)
        chapters.append({"title": build_chapter_title(content, len(chapters) + 1), "content": content})

    return chapters



def build_fallback_review_questions(key_concepts: List[Dict[str, str]], chapters: List[Dict[str, str]]) -> List[str]:
    questions: List[str] = []
    for item in key_concepts[:5]:
        term = item.get("term", "questo concetto")
        questions.append(f"Che cos'è {term} e perché è importante nella lezione?")
    if len(key_concepts) >= 2:
        first_term = key_concepts[0].get("term", "il primo concetto")
        second_term = key_concepts[1].get("term", "il secondo concetto")
        questions.append(f"Come si collegano {first_term} e {second_term} nel ragionamento della lezione?")
    for chapter in chapters[:3]:
        title = chapter.get("title", "questa sezione")
        questions.append(f"Quali sono le idee principali della sezione \"{title}\"?")
    return questions[:8]



def normalize_study_pack(raw_pack: Dict, cleaned_text: str, fallback_paragraphs: List[str]) -> Dict:
    pack = {
        "summary": [],
        "key_concepts": [],
        "chapters": [],
        "glossary": [],
        "review_questions": [],
    }
    if isinstance(raw_pack.get("summary"), list):
        pack["summary"] = [normalize_spaces(str(x)) for x in raw_pack.get("summary", []) if normalize_spaces(str(x))]
    if isinstance(raw_pack.get("key_concepts"), list):
        for item in raw_pack.get("key_concepts", []):
            if isinstance(item, dict):
                term = normalize_spaces(str(item.get("term", "")))
                explanation = normalize_spaces(str(item.get("explanation", "")))
                if term:
                    pack["key_concepts"].append({"term": term, "explanation": explanation})
    if isinstance(raw_pack.get("chapters"), list):
        for item in raw_pack.get("chapters", []):
            if isinstance(item, dict):
                title = normalize_spaces(str(item.get("title", "")))
                content = str(item.get("content", "")).strip()
                if title and content:
                    pack["chapters"].append({"title": title, "content": content})
    if isinstance(raw_pack.get("glossary"), list):
        for item in raw_pack.get("glossary", []):
            if isinstance(item, dict):
                term = normalize_spaces(str(item.get("term", "")))
                definition = normalize_spaces(str(item.get("definition", "")))
                if term:
                    pack["glossary"].append({"term": term, "definition": definition})
    if isinstance(raw_pack.get("review_questions"), list):
        pack["review_questions"] = [
            normalize_spaces(str(x)) for x in raw_pack.get("review_questions", []) if normalize_spaces(str(x))
        ]

    if not pack["summary"]:
        pack["summary"] = build_extract_summary(cleaned_text, max_sentences=7)

    if not pack["key_concepts"]:
        sentences = sentence_split(cleaned_text)
        keywords = extract_keywords(cleaned_text, top_n=8)
        for kw in keywords:
            evidence = find_evidence_sentence(kw, sentences)
            pack["key_concepts"].append(
                {
                    "term": kw.capitalize(),
                    "explanation": evidence or "Concetto ricorrente nella lezione.",
                }
            )

    if not pack["chapters"]:
        pack["chapters"] = build_fallback_chapters(fallback_paragraphs)

    if not pack["glossary"]:
        for item in pack["key_concepts"][:6]:
            pack["glossary"].append({"term": item["term"], "definition": item["explanation"]})

    if not pack["review_questions"]:
        pack["review_questions"] = build_fallback_review_questions(pack["key_concepts"], pack["chapters"])

    return pack



def highlight_terms_in_markdown(text: str, terms: List[str]) -> str:
    highlighted = text
    unique_terms = []
    seen = set()
    for term in terms:
        key = term.lower().strip()
        if key and key not in seen:
            unique_terms.append(term)
            seen.add(key)

    for term in sorted(unique_terms, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(term)})\b", flags=re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted



def build_markdown_package(title: str, cleaned_text: str, study_pack: Dict) -> str:
    md: List[str] = [f"# {title}", ""]

    md.append("## Sintesi")
    md.extend(f"- {item}" for item in study_pack.get("summary", []))
    md.append("")

    md.append("## Concetti chiave")
    for item in study_pack.get("key_concepts", []):
        md.append(f"### {item['term']}")
        md.append(item.get("explanation", ""))
        md.append("")

    md.append("## Capitoli")
    for chapter in study_pack.get("chapters", []):
        md.append(f"### {chapter['title']}")
        md.append(chapter["content"])
        md.append("")

    if study_pack.get("glossary"):
        md.append("## Glossario")
        for item in study_pack["glossary"]:
            md.append(f"- **{item['term']}**: {item.get('definition', '')}")
        md.append("")

    if study_pack.get("review_questions"):
        md.append("## Domande di ripasso")
        for idx, question in enumerate(study_pack["review_questions"], start=1):
            md.append(f"{idx}. {question}")
        md.append("")

    md.append("## Testo completo pulito")
    md.append(cleaned_text)
    md.append("")
    return "\n".join(md)


# -----------------------------
# RTF export
# -----------------------------
def rtf_escape(text: str) -> str:
    parts: List[str] = []
    for ch in text:
        if ch == "\\":
            parts.append(r"\\")
        elif ch == "{":
            parts.append(r"\{")
        elif ch == "}":
            parts.append(r"\}")
        elif ch == "\n":
            parts.append(r"\par " + "\n")
        else:
            code = ord(ch)
            if code > 127:
                signed = code if code <= 32767 else code - 65536
                parts.append(f"\\u{signed}?")
            else:
                parts.append(ch)
    return "".join(parts)



def build_rtf_document(title: str, cleaned_text: str, study_pack: Dict) -> bytes:
    rtf: List[str] = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Helvetica;}}", "\n"]
    rtf.append(r"\fs32\b " + rtf_escape(title) + r"\b0\fs24\par\par " + "\n")

    def add_heading(text: str):
        rtf.append(r"\b " + rtf_escape(text) + r"\b0\par " + "\n")

    add_heading("Sintesi")
    for item in study_pack.get("summary", []):
        rtf.append(r"\bullet " + rtf_escape(item) + r"\par " + "\n")
    rtf.append(r"\par " + "\n")

    add_heading("Concetti chiave")
    for item in study_pack.get("key_concepts", []):
        rtf.append(r"\b " + rtf_escape(item["term"]) + r"\b0\par " + "\n")
        rtf.append(rtf_escape(item.get("explanation", "")) + r"\par\par " + "\n")

    add_heading("Capitoli")
    for chapter in study_pack.get("chapters", []):
        rtf.append(r"\b " + rtf_escape(chapter["title"]) + r"\b0\par " + "\n")
        for paragraph in split_paragraphs(chapter["content"]):
            rtf.append(rtf_escape(paragraph) + r"\par\par " + "\n")

    if study_pack.get("glossary"):
        add_heading("Glossario")
        for item in study_pack["glossary"]:
            rtf.append(r"\b " + rtf_escape(item["term"]) + r"\b0: " + rtf_escape(item.get("definition", "")) + r"\par " + "\n")
        rtf.append(r"\par " + "\n")

    if study_pack.get("review_questions"):
        add_heading("Domande di ripasso")
        for idx, question in enumerate(study_pack["review_questions"], start=1):
            rtf.append(rtf_escape(f"{idx}. {question}") + r"\par " + "\n")
        rtf.append(r"\par " + "\n")

    add_heading("Testo completo pulito")
    for paragraph in split_paragraphs(cleaned_text.replace(". ", ".\n\n")):
        rtf.append(rtf_escape(paragraph) + r"\par\par " + "\n")

    rtf.append("}")
    return "".join(rtf).encode("utf-8")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sbobinatore lezioni", page_icon="🧠", layout="wide")
st.title("🧠 Sbobinatore lezioni in locale")
st.caption(
    "Carica una registrazione, trascrivi in locale con Whisper e ottieni testo completo, sintesi, concetti chiave e export RTF."
)

apple_mac = is_apple_silicon_mac()

with st.sidebar:
    st.header("Impostazioni")
    backend = st.selectbox(
        "Backend trascrizione",
        ["auto", "mlx-whisper", "faster-whisper"] if apple_mac else ["faster-whisper"],
        index=0,
        help="Su Apple Silicon, auto prova MLX e ripiega su faster-whisper se serve.",
    )
    model_size = st.selectbox(
        "Modello Whisper",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(choose_default_model()),
        help="Su Mac Apple Silicon il preset consigliato è mlx-community/whisper-large-v3-turbo.",
    )
    language = st.selectbox("Lingua", ["auto", "it", "en", "es", "fr", "de"], index=0)
    device_options = ["cpu"] if apple_mac else ["cpu", "cuda"]
    device = st.selectbox("Device faster-whisper", device_options, index=0)
    compute_type = st.selectbox("Compute type faster-whisper", ["int8", "float16", "float32"], index=0)
    vad_filter = st.checkbox("Filtra silenzi / pause", value=True)

    st.divider()
    st.subheader("Pacchetto studio")
    summary_mode = st.selectbox("Livello sintesi", ["breve", "standard", "dettagliata"], index=1)
    include_glossary = st.checkbox("Includi glossario", value=True)
    include_review_questions = st.checkbox("Includi domande di ripasso", value=True)

    st.divider()
    st.subheader("Ollama (opzionale)")
    use_ollama = st.checkbox("Usa Ollama per migliorare sintesi e capitoli", value=False)
    enhance_notes_ai = st.checkbox("AI: migliora anche gli appunti finali", value=False)
    ollama_base_url = st.text_input("Ollama URL", value="http://localhost:11434")

    ollama_is_online = use_ollama and ollama_available(ollama_base_url)
    ollama_models = list_ollama_models(ollama_base_url) if ollama_is_online else []

    if use_ollama and not ollama_is_online:
        st.caption("⚠️ Ollama non raggiungibile al momento.")

    ollama_model = st.selectbox(
        "Modello Ollama",
        ollama_models if ollama_models else ["nessuno trovato"],
        index=0,
        help="Consigliati su Mac Apple Silicon: qwen3:30b come principale, gemma3:12b come fallback più leggero.",
    )

st.markdown(
    """
    **Output:**
    - testo completo pulito;
    - sintesi della lezione;
    - concetti chiave evidenziati;
    - capitoli automatici;
    - glossario e domande di ripasso;
    - export in **RTF**, Markdown e TXT.
    """
)

uploaded = st.file_uploader(
    "Carica la lezione",
    type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm", "aac", "flac"],
    accept_multiple_files=False,
)

lesson_title = st.text_input("Titolo lezione", value="Lezione")

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

col1, col2 = st.columns([1, 3])
with col1:
    start_btn = st.button("Avvia elaborazione", type="primary", use_container_width=True)
with col2:
    st.caption(
        "Sul tuo Mac Apple Silicon lascia pure backend=auto e modello=mlx-community/whisper-large-v3-turbo. "
        "Ollama serve soprattutto per avere capitoli, concetti e glossario migliori."
    )

if start_btn:
    if not uploaded:
        st.error("Carica prima un file audio o video.")
        st.stop()

    suffix = os.path.splitext(uploaded.name)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        with st.status("Sto elaborando la lezione...", expanded=True) as status:
            st.write("1/3 Trascrizione locale con Whisper.")
            segments, meta = transcribe_file(
                tmp_path,
                backend=backend,
                model_size=model_size,
                language=language,
                device=device,
                compute_type=compute_type,
                vad_filter=vad_filter,
            )
            st.write(f"Segmenti ottenuti: {len(segments)}")

            raw_text = normalize_spaces(" ".join(s.get("text", "") for s in segments if s.get("text")))
            cleaned_text = clean_transcript_text(raw_text)
            paragraphs = group_segments_to_paragraphs(segments)

            st.write("2/3 Costruzione del pacchetto studio.")
            ollama_used = False
            clipped_for_ollama = False
            raw_pack: Dict = {}

            if use_ollama:
                if not ollama_is_online:
                    st.warning("Ollama non raggiungibile. Continuo con la generazione locale base.")
                elif not ollama_models:
                    st.warning("Nessun modello Ollama trovato. Continuo con la generazione locale base.")
                else:
                    st.write("Uso Ollama per migliorare sintesi, concetti chiave, capitoli e domande di ripasso.")
                    try:
                        raw_pack, clipped_for_ollama = call_ollama_study_pack(
                            model=ollama_model,
                            title=lesson_title,
                            transcript=cleaned_text,
                            base_url=ollama_base_url,
                            summary_mode=summary_mode,
                        )
                        ollama_used = True
                    except Exception as ollama_exc:
                        st.warning(
                            "Ollama ha restituito un output non valido. "
                            "Continuo con la generazione locale base.\n\n"
                            f"Dettaglio: {ollama_exc}"
                        )

            study_pack = normalize_study_pack(raw_pack=raw_pack, cleaned_text=cleaned_text, fallback_paragraphs=paragraphs)

            if not include_glossary:
                study_pack["glossary"] = []
            if not include_review_questions:
                study_pack["review_questions"] = []

            st.write("3/3 Preparazione export e anteprima.")
            keywords = [item["term"] for item in study_pack.get("key_concepts", [])]
            highlighted_text_md = highlight_terms_in_markdown(cleaned_text, keywords)
            markdown_doc = build_markdown_package(lesson_title, cleaned_text, study_pack)
            rtf_bytes = build_rtf_document(lesson_title, cleaned_text, study_pack)
            raw_txt = raw_text
            meta["study_pack_generator"] = "ollama" if ollama_used else "heuristic"
            meta["ollama_clipped_input"] = clipped_for_ollama
            meta["ollama_model"] = ollama_model if use_ollama and ollama_models else None
            meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
            status.update(label="Pacchetto studio pronto", state="complete", expanded=False)

        enhanced_notes_md = ""
        if use_ollama and enhance_notes_ai and ollama_is_online and ollama_models:
            try:
                enhanced_notes_md = call_ollama_enhance_notes(
                    model=ollama_model,
                    title=lesson_title,
                    study_pack=study_pack,
                    cleaned_text=cleaned_text,
                    base_url=ollama_base_url,
                )
            except Exception as notes_exc:
                st.warning(f"Impossibile migliorare automaticamente gli appunti con AI: {notes_exc}")

        st.session_state["last_result"] = {
            "lesson_title": lesson_title,
            "segments": segments,
            "meta": meta,
            "study_pack": study_pack,
            "highlighted_text_md": highlighted_text_md,
            "markdown_doc": markdown_doc,
            "rtf_bytes": rtf_bytes,
            "raw_txt": raw_txt,
            "meta_json": meta_json,
            "enhanced_notes_md": enhanced_notes_md,
        }

    except Exception as e:
        st.exception(e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

result = st.session_state.get("last_result")
if result:
    st.success("Fatto. Hai il testo completo, la sintesi e i file esportabili.")
    meta = result["meta"]
    study_pack = result["study_pack"]
    segments = result["segments"]

    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Segmenti", len(segments))
    info2.metric("Lingua rilevata", meta.get("language") or "n/d")
    info3.metric("Durata", format_ts(meta.get("duration")) if meta.get("duration") else "n/d")
    info4.metric("Pacchetto studio", meta.get("study_pack_generator", "n/d"))

    if meta.get("ollama_clipped_input"):
        st.info(
            "Per Ollama è stata inviata solo una parte iniziale della trascrizione per contenere tempi e memoria. "
            "Il testo completo pulito resta comunque disponibile integralmente nell'export."
        )

    tab1, tab2, tab3, tab4 = st.tabs(["Pacchetto studio", "Testo completo", "Trascrizione fedele", "Metadati"])
    with tab1:
        st.subheader("Sintesi")
        for item in study_pack.get("summary", []):
            st.markdown(f"- {item}")

        st.subheader("Concetti chiave")
        for item in study_pack.get("key_concepts", []):
            st.markdown(f"**{item['term']}** — {item.get('explanation', '')}")

        st.subheader("Capitoli")
        for chapter in study_pack.get("chapters", []):
            with st.expander(chapter["title"], expanded=False):
                st.write(chapter["content"])

        if study_pack.get("glossary"):
            st.subheader("Glossario")
            for item in study_pack["glossary"]:
                st.markdown(f"- **{item['term']}**: {item.get('definition', '')}")

        if study_pack.get("review_questions"):
            st.subheader("Domande di ripasso")
            for idx, question in enumerate(study_pack["review_questions"], start=1):
                st.markdown(f"{idx}. {question}")

        if result.get("enhanced_notes_md"):
            st.subheader("Appunti migliorati con AI")
            st.markdown(result["enhanced_notes_md"])

    with tab2:
        st.subheader("Testo completo pulito con concetti evidenziati")
        st.markdown(result["highlighted_text_md"])

    with tab3:
        st.subheader("Trascrizione fedele")
        st.text_area("Testo grezzo", result["raw_txt"], height=420)

    with tab4:
        st.code(result["meta_json"], language="json")

    base_name = slugify(result["lesson_title"])
    st.download_button(
        "Scarica pacchetto studio .rtf",
        data=result["rtf_bytes"],
        file_name=f"{base_name}_studio.rtf",
        mime="application/rtf",
        use_container_width=True,
    )
    st.download_button(
        "Scarica pacchetto studio .md",
        data=result["markdown_doc"].encode("utf-8"),
        file_name=f"{base_name}_studio.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if result.get("enhanced_notes_md"):
        st.download_button(
            "Scarica appunti migliorati .md",
            data=result["enhanced_notes_md"].encode("utf-8"),
            file_name=f"{base_name}_appunti_ai.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.download_button(
        "Scarica trascrizione .txt",
        data=result["raw_txt"].encode("utf-8"),
        file_name=f"{base_name}_trascrizione.txt",
        mime="text/plain",
        use_container_width=True,
    )

with st.expander("Istruzioni di installazione"):
    st.markdown(
        """
        ```bash
        brew install ffmpeg
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        streamlit run app.py
        ```

        Note:
        - su Mac Apple Silicon l'impostazione più comoda è `backend=auto`;
        - per il pacchetto studio avanzato installa **Ollama** e un modello, ad esempio `qwen3:30b` oppure `gemma3:12b`;
        - la trascrizione resta in locale.
        """
    )
