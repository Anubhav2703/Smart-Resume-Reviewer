# app.py — Smart Resume Reviewer (corrected full file)
# ---------------------------------------------------
# Features included:
# - Resume upload (PDF/text) and paste
# - Optional Job Description upload/paste
# - LLM-powered review integration points (OpenAI client supported)
# - Robust JSON parsing and normalization of LLM output
# - Section-wise feedback, keyword gaps, duplicate/vague phrases
# - Scoring normalization to satisfy Pydantic schema
# - Improved resume generation (from LLM or synthesized from suggestions)
# - Improved resume displayed and editable in UI
# - Side-by-side comparison (Original vs Improved) + unified diff
# - Download improved resume as PDF (sanitized for FPDF / latin-1)
# - Session-state caching of last review
#
# Notes:
# - If OPENAI_API_KEY is set and OpenAI SDK is installed, real LLM calls will be made.
# - If not configured, the app will use a demo fallback response so UI can be tested.
#
# Requirements:
# pip install streamlit pydantic fpdf pymupdf pdfplumber PyPDF2 openai
#
# Run:
# streamlit run app.py
# ---------------------------------------------------

import os
import io
import json
import re
import difflib
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

from typing import Optional, List, Dict, Any

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from fpdf import FPDF

# Optional PDF parsing libraries
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Optional OpenAI client
_openai_available = False
try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

# Initialize OpenAI client if available & key set
_openai_client = None
if _openai_available and os.getenv("OPENAI_API_KEY"):
    try:
        _openai_client = OpenAI()
    except Exception:
        _openai_client = None

# ---- App configuration ----
APP_TITLE = "Smart Resume Reviewer"
PRIVACY_NOTE = (
    "We process resumes in-memory for this session only. We do not persist uploaded files. "
    "If you use an external LLM provider, content is sent to that provider according to their policy."
)
DEFAULT_MODEL = "gpt-4o-mini"
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4"]

# ---- Pydantic schemas ----
class Scoring(BaseModel):
    relevance: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)
    impact: int = Field(ge=0, le=10)
    ats: int = Field(ge=0, le=10)

class SectionFeedback(BaseModel):
    Education: List[str] = []
    Experience: List[str] = []
    Skills: List[str] = []
    Projects: List[str] = []
    Certifications: List[str] = []
    SummaryOrObjective: List[str] = []
    Other: List[str] = []

class ReviewResponse(BaseModel):
    overall_summary: str
    keyword_gaps: List[str]
    duplicate_or_vague_phrases: List[str]
    suggestions: List[str]
    section_feedback: SectionFeedback
    scoring: Scoring
    improved_resume: Optional[str] = None

# ---- LLM system prompt ----
LLM_SYSTEM_PROMPT = (
    "You are a meticulous resume coach. Evaluate a candidate's resume for a specific target role. "
    "Optionally use the provided job description (JD) to tailor feedback. Provide concrete, actionable, "
    "section-wise feedback and return ONLY a valid JSON object matching the schema described below:\n"
    " - overall_summary: string\n"
    " - keyword_gaps: list of strings\n"
    " - duplicate_or_vague_phrases: list of strings\n"
    " - suggestions: list of concise strings (actionable bullets)\n"
    " - section_feedback: object with lists for Education, Experience, Skills, Projects, Certifications, SummaryOrObjective, Other\n"
    " - scoring: object with integer values 0-10 for keys relevance, clarity, impact, ats\n"
    " - improved_resume: optional string containing a full improved resume draft with headings and bullets\n"
    "Make output concise and actionable. Return strictly valid JSON (no extra text)."
)

# ---- Utilities: PDF/text extraction ----
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Try multiple parsers to extract text from PDF bytes (best-effort)."""
    text = ""
    # Try PyMuPDF
    if fitz:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    page_text = page.get_text() or ""
                    text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception:
            pass

    # Try pdfplumber
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception:
            pass

    # Try PyPDF2
    if PyPDF2:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                page_text = p.extract_text() or ""
                text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception:
            pass

    # Fallback: try decode as UTF-8 ignoring errors
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

def sanitize_text_for_display(t: str) -> str:
    """Light sanitization for UI display (keeps Unicode)."""
    if not t:
        return ""
    s = str(t)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\t+", " ", s)
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---- Sanitize for PDF export (FPDF / latin-1 safe) ----
def sanitize_for_pdf(s: str) -> str:
    """
    Replace problematic Unicode characters with ASCII-safe equivalents,
    ensuring the final string is latin-1 encodable (FPDF uses latin-1).
    """
    if s is None:
        s = ""
    s = str(s)
    replacements = {
        "•": "-", "\u2022": "-",  # bullets -> hyphen
        "–": "-", "—": "-",       # dashes -> hyphen
        "…": "...",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "\u00a0": " ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # collapse multiple spaces
    s = re.sub(r"[ \t]+", " ", s)
    # enforce latin-1: unknown -> '?'
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s

def generate_pdf_bytes_safe(text: str) -> bytes:
    """
    Generate PDF bytes using FPDF. Text is sanitized for latin-1.
    Headings wrapped with ** are rendered bold. Bullets starting with '- ' are used.
    """
    sanitized = sanitize_for_pdf(text)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Improved Resume", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    for raw_line in sanitized.splitlines():
        line = raw_line.rstrip()
        if not line:
            pdf.ln(2)
            continue
        if line.startswith("**") and line.endswith("**"):
            header = line.strip("*").strip()
            pdf.set_font("Arial", "B", 13)
            pdf.multi_cell(0, 8, header)
            pdf.set_font("Arial", size=12)
        elif line.startswith("- "):
            pdf.multi_cell(0, 7, "- " + line[2:].strip())
        else:
            pdf.multi_cell(0, 7, line)
    output_str = pdf.output(dest="S")
    if isinstance(output_str, str):
        output_bytes = output_str.encode("latin-1", "replace")
    else:
        output_bytes = output_str
    return output_bytes

# ---- LLM wrapper ----
def call_llm_api(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Call OpenAI iff configured. Return the raw string response.
    Throws RuntimeError if client isn't available.
    """
    if not _openai_client:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
    resp = _openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ---- Prompt builder ----
def build_user_prompt(resume_text: str, target_role: str, jd_text: Optional[str]) -> str:
    guidance = (
        "Return a single JSON object with keys: overall_summary (string), "
        "keyword_gaps (list), duplicate_or_vague_phrases (list), suggestions (list), "
        "section_feedback (object with lists for Education, Experience, Skills, Projects, Certifications, SummaryOrObjective, Other), "
        "scoring (object with integers 0-10 for relevance, clarity, impact, ats), improved_resume (string, optional)."
    )
    jd_block = f"\nJOB DESCRIPTION:\n{jd_text.strip()}\n" if jd_text else "\n(no JD provided)\n"
    return f"TARGET ROLE: {target_role}\n{jd_block}\nRESUME TEXT:\n{resume_text.strip()}\n\n{guidance}"

# ---- Robust JSON extraction ----
def extract_json_from_raw(raw: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON directly; if fails, extract first {...} block via regex and parse.
    Returns dict or None.
    """
    if not raw or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None

# ---- Synthesize improved resume fallback ----
def synthesize_improved_resume(data: Dict[str, Any], target_role: str) -> str:
    parts = []
    parts.append(f"**Improved Resume — Target Role: {target_role}**\n")
    overall = data.get("overall_summary", "").strip()
    if overall:
        parts.append("**Summary**")
        parts.append(overall)
        parts.append("")
    sf = data.get("section_feedback", {}) or {}
    for sec in ["Experience", "Education", "Skills", "Projects", "Certifications", "SummaryOrObjective", "Other"]:
        items = sf.get(sec, [])
        if items:
            parts.append(f"**{sec}**")
            for it in items:
                parts.append(f"- {it}")
            parts.append("")
    suggestions = data.get("suggestions", []) or []
    if suggestions:
        parts.append("**Suggestions**")
        for s in suggestions:
            parts.append(f"- {s}")
        parts.append("")
    kw = data.get("keyword_gaps", []) or []
    if kw:
        parts.append("**Keyword Gaps (Consider adding)**")
        for k in kw:
            parts.append(f"- {k}")
        parts.append("")
    return "\n".join(parts).strip()

# ---- Streamlit UI ----
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(APP_TITLE)  # simplified heading (no "Detailed")
st.caption(PRIVACY_NOTE)

# Sidebar: settings / LLM toggle
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", MODEL_OPTIONS, index=0)
    gen_improved_toggle = st.checkbox("Generate improved resume draft", value=True)
    st.markdown("---")
    if not _openai_client:
        st.warning("OpenAI client not configured. Demo fallback will be used.")

# Input columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload PDF resume", type=["pdf"])
    resume_text_input = st.text_area("Or paste resume text", height=240, placeholder="Paste resume content…")
with col2:
    st.subheader("Target Role & JD")
    target_role = st.text_input("Target Job Role (e.g., Data Scientist, Product Manager)")
    jd_file = st.file_uploader("Optional: Upload JD (PDF)", type=["pdf"], key="jdpdf")
    jd_text_input = st.text_area("Or paste Job Description text (optional)", height=240, placeholder="Paste JD…")

# Ingest resume text
resume_text_raw = ""
if resume_file:
    try:
        resume_bytes = resume_file.read()
        resume_text_raw = extract_text_from_pdf_bytes(resume_bytes)
    except Exception:
        resume_text_raw = ""
if not resume_text_raw and resume_text_input:
    resume_text_raw = resume_text_input
resume_text = sanitize_text_for_display(resume_text_raw)

# Ingest JD text
jd_text_raw = ""
if jd_file:
    try:
        jd_bytes = jd_file.read()
        jd_text_raw = extract_text_from_pdf_bytes(jd_bytes)
    except Exception:
        jd_text_raw = ""
if not jd_text_raw and jd_text_input:
    jd_text_raw = jd_text_input
jd_text = sanitize_text_for_display(jd_text_raw)

st.divider()
run_review = st.button("▶️ Run Review", type="primary", use_container_width=True)

# Session state for caching last review
if "last_review_data" not in st.session_state:
    st.session_state["last_review_data"] = None
if "last_raw" not in st.session_state:
    st.session_state["last_raw"] = None
if "last_improved_text" not in st.session_state:
    st.session_state["last_improved_text"] = None

if run_review:
    if not resume_text.strip():
        st.error("Please upload a resume or paste text.")
    elif not target_role.strip():
        st.error("Please enter a target job role.")
    else:
        with st.spinner("Analyzing resume with LLM…"):
            try:
                user_prompt = build_user_prompt(resume_text, target_role.strip(), jd_text or None)
                raw = ""
                if _openai_client:
                    try:
                        raw = call_llm_api(model_choice, LLM_SYSTEM_PROMPT, user_prompt)
                    except Exception as e:
                        st.warning(f"LLM call failed: {e}. Using demo fallback.")
                        raw = ""
                else:
                    raw = ""

                st.session_state["last_raw"] = raw

                parsed = extract_json_from_raw(raw) if raw else None

                if not parsed:
                    # Demo fallback structured response for UI testing
                    parsed = {
                        "overall_summary": "Demo: resume needs improvement to better align with the target role.",
                        "keyword_gaps": ["Digital Marketing", "SEO", "Analytics"],
                        "duplicate_or_vague_phrases": ["various marketing things", "outreaching new people"],
                        "suggestions": [
                            "Add measurable achievements with percentages and numbers.",
                            "Include tools like Google Analytics, SEMrush.",
                            "Clarify roles and responsibilities with action verbs."
                        ],
                        "section_feedback": {
                            "Education": ["Include relevant coursework related to marketing."],
                            "Experience": ["Quantify achievements.", "Use action-oriented bullets."],
                            "Skills": ["Prioritize marketing-related skills."],
                            "Projects": [],
                            "Certifications": ["Add Google Analytics certification if available."],
                            "SummaryOrObjective": ["Add a concise summary focusing on marketing goals."],
                            "Other": []
                        },
                        "scoring": {"relevance": 4, "clarity": 5, "impact": 3, "ats": 4},
                        "improved_resume": ""
                    }

                # Normalize scoring
                scoring_data = parsed.get("scoring", {})
                if isinstance(scoring_data, int):
                    parsed["scoring"] = {
                        "relevance": int(scoring_data),
                        "clarity": 7,
                        "impact": 7,
                        "ats": 7
                    }
                elif not isinstance(scoring_data, dict):
                    parsed["scoring"] = {"relevance": 0, "clarity": 0, "impact": 0, "ats": 0}
                else:
                    for key in ["relevance", "clarity", "impact", "ats"]:
                        val = scoring_data.get(key)
                        if not isinstance(val, int):
                            scoring_data[key] = 7
                    parsed["scoring"] = scoring_data

                # Ensure section_feedback lists
                section_feedback = parsed.get("section_feedback", {}) or {}
                for key in ["Education", "Experience", "Skills", "Projects", "Certifications", "SummaryOrObjective", "Other"]:
                    val = section_feedback.get(key, [])
                    if isinstance(val, str):
                        section_feedback[key] = [val]
                    elif not isinstance(val, list):
                        section_feedback[key] = []
                parsed["section_feedback"] = section_feedback

                # Normalize suggestions
                suggestions = parsed.get("suggestions", [])
                if isinstance(suggestions, str):
                    parsed["suggestions"] = [s.strip(" -•") for s in suggestions.splitlines() if s.strip()]

                # Normalize arrays like keyword_gaps
                for arr_key in ["keyword_gaps", "duplicate_or_vague_phrases"]:
                    val = parsed.get(arr_key, [])
                    if isinstance(val, str):
                        parsed[arr_key] = [v.strip() for v in re.split(r"[,\n;]+", val) if v.strip()]
                    elif not isinstance(val, list):
                        parsed[arr_key] = []

                # Build improved resume fallback if empty
                improved_resume_text = parsed.get("improved_resume") or ""
                if not isinstance(improved_resume_text, str):
                    improved_resume_text = ""
                if not improved_resume_text.strip():
                    improved_resume_text = synthesize_improved_resume(parsed, target_role.strip())
                parsed["improved_resume"] = improved_resume_text

                # Try Pydantic validation
                try:
                    review = ReviewResponse(**parsed)
                except ValidationError as ve:
                    st.warning("LLM output did not fully match schema; attempting minimal coercion.")
                    st.exception(ve)
                    # Coerce minimal values and revalidate
                    parsed.setdefault("overall_summary", parsed.get("overall_summary", ""))
                    parsed.setdefault("keyword_gaps", parsed.get("keyword_gaps", []))
                    parsed.setdefault("duplicate_or_vague_phrases", parsed.get("duplicate_or_vague_phrases", []))
                    parsed.setdefault("suggestions", parsed.get("suggestions", []))
                    parsed.setdefault("section_feedback", parsed.get("section_feedback", {}))
                    parsed.setdefault("scoring", parsed.get("scoring", {"relevance":0,"clarity":0,"impact":0,"ats":0}))
                    parsed.setdefault("improved_resume", parsed.get("improved_resume", ""))
                    review = ReviewResponse(**parsed)

                # Save to session
                st.session_state["last_review_data"] = parsed
                st.session_state["last_improved_text"] = parsed["improved_resume"]

                # Display results
                st.success("Review complete.")
                with st.expander("Overall Summary", expanded=True):
                    st.write(review.overall_summary)

                with st.expander("Scores (0–10)", expanded=False):
                    cols = st.columns(4)
                    cols[0].metric("Relevance", review.scoring.relevance)
                    cols[1].metric("Clarity", review.scoring.clarity)
                    cols[2].metric("Impact", review.scoring.impact)
                    cols[3].metric("ATS", review.scoring.ats)

                with st.expander("Keyword Gaps", expanded=False):
                    if review.keyword_gaps:
                        for k in review.keyword_gaps:
                            st.markdown(f"- {k}")
                    else:
                        st.write("No major gaps detected.")

                with st.expander("Duplicate / Vague Phrases", expanded=False):
                    if review.duplicate_or_vague_phrases:
                        for p in review.duplicate_or_vague_phrases:
                            st.markdown(f"- {p}")
                    else:
                        st.write("None detected.")

                with st.expander("Section-wise Feedback", expanded=True):
                    sf_obj = review.section_feedback
                    ordering = [
                        ("Summary / Objective", sf_obj.SummaryOrObjective),
                        ("Experience", sf_obj.Experience),
                        ("Education", sf_obj.Education),
                        ("Skills", sf_obj.Skills),
                        ("Projects", sf_obj.Projects),
                        ("Certifications", sf_obj.Certifications),
                        ("Other", sf_obj.Other),
                    ]
                    for label, items in ordering:
                        if items:
                            st.markdown(f"**{label}**")
                            for it in items:
                                st.markdown(f"- {it}")

                with st.expander("General Suggestions", expanded=False):
                    if review.suggestions:
                        for s in review.suggestions:
                            st.markdown(f"- {s}")
                    else:
                        st.write("No suggestions available.")

                # Improved resume + download
                if gen_improved_toggle:
                    st.divider()
                    st.subheader("Improved Resume (editable)")
                    initial_improved = st.session_state.get("last_improved_text", review.improved_resume or "")
                    edited = st.text_area("Edit improved resume (you can tweak before export)", value=initial_improved, height=360, key="improved_edit")
                    if edited != st.session_state.get("last_improved_text"):
                        st.session_state["last_improved_text"] = edited

                    pdf_bytes = generate_pdf_bytes_safe(edited)
                    st.download_button(
                        "📄 Download Improved Resume (PDF)",
                        data=pdf_bytes,
                        file_name="improved_resume.pdf",
                        mime="application/pdf",
                    )

                # Comparison view
                if resume_text:
                    st.divider()
                    st.subheader("Compare: Original Resume ⇢ Improved Resume")
                    show_diff = st.checkbox("Show unified diff", value=False)
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Original Resume**")
                        st.text_area("Original Resume (read-only)", value=resume_text, height=360, key="orig_area")
                    with right:
                        st.markdown("**Improved Resume**")
                        improved_content = st.session_state.get("last_improved_text", review.improved_resume or "")
                        new_improved = st.text_area("Improved (editable)", value=improved_content, height=360, key="improved_edit2")
                        if new_improved != improved_content:
                            st.session_state["last_improved_text"] = new_improved
                            improved_content = new_improved

                        pdf_bytes2 = generate_pdf_bytes_safe(improved_content)
                        st.download_button(
                            "📄 Download Current Improved (PDF)",
                            data=pdf_bytes2,
                            file_name="improved_resume_current.pdf",
                            mime="application/pdf",
                        )

                    if show_diff:
                        st.divider()
                        st.subheader("Unified Diff (Original → Improved)")
                        orig_lines = resume_text.splitlines(keepends=False)
                        new_lines = improved_content.splitlines(keepends=False)
                        diff = difflib.unified_diff(orig_lines, new_lines, lineterm="")
                        diff_text = "\n".join(diff)
                        if diff_text.strip():
                            st.code(diff_text)
                        else:
                            st.write("No differences detected.")

            except Exception as e:
                st.error("Something went wrong while running the review.")
                st.exception(e)

# If previous review exists allow reopen
if st.session_state.get("last_review_data") and not run_review:
    st.divider()
    st.info("A previous review exists in this session.")
    if st.button("Open last review / comparison"):
        data = st.session_state.get("last_review_data")
        try:
            review = ReviewResponse(**data)
            st.success("Loaded previous review.")
            with st.expander("Overall Summary (previous)", expanded=False):
                st.write(review.overall_summary)
            st.subheader("Improved Resume (previous)")
            prev_improved = st.session_state.get("last_improved_text", review.improved_resume or "")
            st.text_area("Improved Resume (editable, previous)", value=prev_improved, height=360, key="prev_improved_area")
            pdf_prev = generate_pdf_bytes_safe(prev_improved)
            st.download_button(
                "📄 Download Previous Improved (PDF)",
                data=pdf_prev,
                file_name="improved_resume_previous.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error("Could not load previous review.")
            st.exception(e)

