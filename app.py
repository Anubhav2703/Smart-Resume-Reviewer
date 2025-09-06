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

# No tips section — intentionally removed per user request

# End of file

# # app.py — Smart Resume Reviewer (detailed, full-featured)
# # ------------------------------------------------------
# # Features:
# # - Upload resume (PDF) or paste text
# # - Optional upload/paste Job Description (JD)
# # - LLM-powered review (OpenAI client expected; fallback demo if not configured)
# # - Robust JSON parsing and normalization of LLM output
# # - Section-wise feedback, keyword gaps, duplicate/vague phrases
# # - Scoring normalization to satisfy Pydantic schema
# # - Improved resume generation (from LLM or synthesized from suggestions)
# # - Improved resume shown and editable in UI
# # - Side-by-side comparison (original vs improved) with unified diff
# # - Download improved resume as PDF (safe for latin-1 / sanitized)
# # - Session-state caching of last review
# #
# # Requirements:
# #   pip install streamlit pydantic fpdf pymupdf pdfplumber openai PyPDF2
# # (pdf libraries optional; code falls back gracefully)
# #
# # Run:
# #   streamlit run app.py
# #
# # Make sure to set OPENAI_API_KEY if you want real LLM responses:
# #   export OPENAI_API_KEY="sk-..."
# # ------------------------------------------------------

# import os
# import io
# import json
# import re
# import difflib
# from typing import Optional, List, Dict, Any

# import streamlit as st
# from pydantic import BaseModel, Field, ValidationError

# # PDF generation (fpdf, using latin-1 safe output after sanitization)
# from fpdf import FPDF

# # Optional PDF parsers
# try:
#     import fitz  # PyMuPDF
# except Exception:
#     fitz = None

# try:
#     import pdfplumber
# except Exception:
#     pdfplumber = None

# # Optional PyPDF2 (more robust PDF text extraction fallback)
# try:
#     import PyPDF2
# except Exception:
#     PyPDF2 = None

# # Optional OpenAI client
# _openai_available = False
# try:
#     from openai import OpenAI  # new official SDK imports may vary
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # ---- App configuration ----
# APP_TITLE = "Smart Resume Reviewer (Detailed)"
# PRIVACY_NOTE = (
#     "We process resumes in-memory for this session only. We do not persist uploaded files. "
#     "If you use an external LLM provider, content is sent to that provider according to their policy."
# )
# DEFAULT_MODEL = "gpt-4o-mini"
# MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4"]  # adjust to available models

# # ---- Pydantic Schemas ----
# class Scoring(BaseModel):
#     relevance: int = Field(ge=0, le=10, description="Fit to target role/JD")
#     clarity: int = Field(ge=0, le=10, description="Readability & organization")
#     impact: int = Field(ge=0, le=10, description="Achievement focus & outcomes")
#     ats: int = Field(ge=0, le=10, description="Keyword/format readiness for ATS")


# class SectionFeedback(BaseModel):
#     Education: List[str] = []
#     Experience: List[str] = []
#     Skills: List[str] = []
#     Projects: List[str] = []
#     Certifications: List[str] = []
#     SummaryOrObjective: List[str] = []
#     Other: List[str] = []


# class ReviewResponse(BaseModel):
#     overall_summary: str
#     keyword_gaps: List[str]
#     duplicate_or_vague_phrases: List[str]
#     suggestions: List[str]
#     section_feedback: SectionFeedback
#     scoring: Scoring
#     improved_resume: Optional[str] = None


# # ---- OpenAI client init (if available & env set) ----
# _openai_client = None
# if _openai_available and os.getenv("OPENAI_API_KEY"):
#     try:
#         _openai_client = OpenAI()
#     except Exception:
#         _openai_client = None

# # ---- LLM system prompt (guides output format) ----
# LLM_SYSTEM_PROMPT = (
#     "You are a meticulous resume coach. Evaluate a candidate's resume for a specific target role. "
#     "Optionally use the provided job description (JD) to tailor feedback. Provide concrete, actionable, "
#     "section-wise feedback. Prioritize measurable achievements, relevant keywords, and clarity. "
#     "Output only a valid JSON object (no surrounding explanation) with keys:\n"
#     "overall_summary (string),\n"
#     "keyword_gaps (list of strings),\n"
#     "duplicate_or_vague_phrases (list of strings),\n"
#     "suggestions (list of concise bullet strings),\n"
#     "section_feedback (object with lists for Education, Experience, Skills, Projects, Certifications, SummaryOrObjective, Other),\n"
#     "scoring (object with integer values 0-10 for relevance, clarity, impact, ats),\n"
#     "improved_resume (string, optional — a full improved resume draft with headings and bullets).\n"
# )

# # ---- Utility functions: PDF/text extraction & sanitization ----
# def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
#     """Try PyMuPDF, then pdfplumber, then PyPDF2 — best-effort text extraction."""
#     text = ""

#     if fitz:
#         try:
#             with fitz.open(stream=file_bytes, filetype="pdf") as doc:
#                 for page in doc:
#                     text += page.get_text() or ""
#             if text.strip():
#                 return text.strip()
#         except Exception:
#             pass

#     if pdfplumber:
#         try:
#             with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text() or ""
#                     text += page_text + "\n"
#             if text.strip():
#                 return text.strip()
#         except Exception:
#             pass

#     if PyPDF2:
#         try:
#             reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
#             for p in reader.pages:
#                 text += (p.extract_text() or "") + "\n"
#             if text.strip():
#                 return text.strip()
#         except Exception:
#             pass

#     # Last resort: try naive decode
#     try:
#         return file_bytes.decode("utf-8", errors="ignore").strip()
#     except Exception:
#         return ""

# def sanitize_text_for_display(t: str) -> str:
#     """Light sanitization for showing in UI (preserve Unicode)."""
#     if not t:
#         return ""
#     t = t.replace("\u00a0", " ")
#     t = re.sub(r"\t+", " ", t)
#     t = re.sub(r"\r\n", "\n", t)
#     t = re.sub(r"\n{3,}", "\n\n", t)
#     return t.strip()

# def sanitize_for_pdf(s: str) -> str:
#     """
#     Convert / replace characters that are problematic for FPDF (latin-1)
#     into safe ASCII equivalents. Returns a latin-1-safe string.
#     """
#     if s is None:
#         s = ""
#     s = str(s)
#     replacements = {
#         "\u2022": "-",  # bullet
#         "•": "-",       # bullet
#         "–": "-",       # en dash
#         "—": "-",       # em dash
#         "…": "...",     # ellipsis
#         "“": '"', "”": '"', "‘": "'", "’": "'",
#         "\u00a0": " ",
#     }
#     for k, v in replacements.items():
#         s = s.replace(k, v)
#     # Collapse multiple spaces
#     s = re.sub(r"[ \t]+", " ", s)
#     # Ensure latin-1: replace any non-encodable char with '?'
#     s = s.encode("latin-1", "replace").decode("latin-1")
#     return s

# def generate_pdf_bytes_safe(text: str) -> bytes:
#     """
#     Generate PDF bytes using FPDF and sanitized text (latin-1 safe).
#     This keeps formatting simple: headings in bold if wrapped in **, bullets starting with '- ' are rendered.
#     """
#     sanitized = sanitize_for_pdf(text)
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=12)
#     pdf.add_page()
#     # Title
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "Improved Resume", ln=True, align="C")
#     pdf.ln(4)
#     pdf.set_font("Arial", size=12)
#     for raw_line in sanitized.splitlines():
#         line = raw_line.rstrip()
#         if not line:
#             pdf.ln(2)
#             continue
#         if line.startswith("**") and line.endswith("**"):
#             # treat as section header
#             header = line.strip("*").strip()
#             pdf.set_font("Arial", "B", 13)
#             pdf.multi_cell(0, 8, header)
#             pdf.set_font("Arial", size=12)
#         elif line.startswith("- "):
#             # bullet (render with ASCII dash; sanitize already replaced unicode bullets)
#             pdf.multi_cell(0, 7, "- " + line[2:].strip())
#         else:
#             pdf.multi_cell(0, 7, line)
#     # Export as bytes (S returns a str in pyfpdf — encode to latin-1)
#     output_str = pdf.output(dest="S")
#     if isinstance(output_str, str):
#         output_bytes = output_str.encode("latin-1", "replace")
#     else:
#         output_bytes = output_str
#     return output_bytes

# # ---- LLM call wrapper (OpenAI) ----
# def call_llm_api(model: str, system_prompt: str, user_prompt: str) -> str:
#     """
#     Calls OpenAI via the new SDK wrapper if configured.
#     Returns raw text output (string). Caller should robustly parse JSON from it.
#     """
#     if not _openai_client:
#         raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY environment variable.")
#     # Using Chat Completions via the OpenAI client wrapper — API signature may vary across SDK versions.
#     resp = _openai_client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         temperature=0.2,
#     )
#     # Extract text content
#     content = resp.choices[0].message.content
#     return content

# # ---- Prompt builder ----
# def build_user_prompt(resume_text: str, target_role: str, jd_text: Optional[str]) -> str:
#     guidance = (
#         "You must return a single JSON object (no surrounding explanation) with keys:\n"
#         "overall_summary (str), keyword_gaps (list[str]), duplicate_or_vague_phrases (list[str]),\n"
#         "suggestions (list[str]), section_feedback (dict of lists), scoring (dict of ints 0-10 for relevance, clarity, impact, ats),\n"
#         "improved_resume (string, optional; full resume draft with headings and bullets).\n"
#         "Be concise and actionable."
#     )
#     jd_block = f"\nJOB DESCRIPTION:\n{jd_text.strip()}\n" if jd_text else "\n(no JD provided)\n"
#     return f"TARGET ROLE: {target_role}\n{jd_block}\nRESUME TEXT:\n{resume_text.strip()}\n\n{guidance}"

# # ---- Helper: robust JSON extraction from LLM output ----
# def extract_json_from_raw(raw: str) -> Optional[Dict[str, Any]]:
#     """
#     Attempts to parse JSON from LLM raw output. Tries:
#     1) json.loads(raw) directly
#     2) regex find first {...} block and parse that
#     Returns dict or None.
#     """
#     if not raw or not raw.strip():
#         return None
#     # direct parse
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         pass
#     # Try to extract first JSON object {...}
#     m = re.search(r"\{.*\}", raw, re.DOTALL)
#     if m:
#         try:
#             return json.loads(m.group(0))
#         except json.JSONDecodeError:
#             return None
#     return None

# # ---- Utility: synthesize improved resume from review data (fallback) ----
# def synthesize_improved_resume(data: Dict[str, Any], target_role: str) -> str:
#     """
#     Build a readable improved resume-like markdown text from the fields in data.
#     Only used when LLM didn't return a full improved_resume.
#     """
#     parts = []
#     # Title / header
#     parts.append(f"**Improved Resume — Target Role: {target_role}**\n")
#     # Overall summary
#     overall = data.get("overall_summary", "").strip()
#     if overall:
#         parts.append("**Summary**")
#         parts.append(overall)
#         parts.append("")
#     # Section feedback mapping — each becomes a section with bullets
#     sf = data.get("section_feedback", {}) or {}
#     for sec_label in ["Experience", "Education", "Skills", "Projects", "Certifications", "SummaryOrObjective", "Other"]:
#         items = sf.get(sec_label) or []
#         if items:
#             parts.append(f"**{sec_label}**")
#             for it in items:
#                 parts.append(f"- {it}")
#             parts.append("")
#     # Suggestions
#     suggestions = data.get("suggestions", []) or []
#     if suggestions:
#         parts.append("**Suggestions**")
#         for s in suggestions:
#             parts.append(f"- {s}")
#         parts.append("")
#     # Keyword gaps
#     kw = data.get("keyword_gaps", []) or []
#     if kw:
#         parts.append("**Keyword Gaps (Consider adding)**")
#         for k in kw:
#             parts.append(f"- {k}")
#         parts.append("")
#     return "\n".join(parts).strip()

# # ---- Streamlit UI and main flow ----
# st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
# st.title(APP_TITLE)
# st.caption(PRIVACY_NOTE)

# # Sidebar: settings & uploads
# with st.sidebar:
#     st.header("Settings")
#     model_choice = st.selectbox("Model", MODEL_OPTIONS, index=0)
#     gen_improved_toggle = st.checkbox("Generate improved resume draft", True)
#     st.markdown("---")
#     st.write("LLM:")
#     if not _openai_client:
#         st.warning("OPENAI client not configured. App will use demo fallback output instead.")
#     st.markdown("---")

# col_left, col_right = st.columns(2)

# with col_left:
#     st.subheader("Resume Input")
#     uploaded_pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
#     resume_text_area = st.text_area("Or paste resume text", height=300, placeholder="Paste resume here...")
# with col_right:
#     st.subheader("Target Role / Job Description")
#     target_role = st.text_input("Target job role (e.g., Data Scientist)", placeholder="e.g., Product Manager")
#     jd_pdf = st.file_uploader("Optional: upload JD (PDF)", type=["pdf"], key="jd_pdf")
#     jd_text_area = st.text_area("Or paste job description text (optional)", height=300)

# # Read and extract resume text
# resume_text_raw = ""
# if uploaded_pdf:
#     try:
#         file_bytes = uploaded_pdf.read()
#         resume_text_raw = extract_text_from_pdf_bytes(file_bytes)
#     except Exception:
#         resume_text_raw = ""
# if not resume_text_raw and resume_text_area:
#     resume_text_raw = resume_text_area

# resume_text = sanitize_text_for_display(resume_text_raw)

# # Read JD text
# jd_text_raw = ""
# if jd_pdf:
#     try:
#         jd_bytes = jd_pdf.read()
#         jd_text_raw = extract_text_from_pdf_bytes(jd_bytes)
#     except Exception:
#         jd_text_raw = ""
# if not jd_text_raw and jd_text_area:
#     jd_text_raw = jd_text_area
# jd_text = sanitize_text_for_display(jd_text_raw)

# st.divider()
# run = st.button("▶️ Run Review", type="primary", use_container_width=True)

# # Session state to store last review (so user can compare without rerunning)
# if "last_review" not in st.session_state:
#     st.session_state["last_review"] = None
# if "last_raw" not in st.session_state:
#     st.session_state["last_raw"] = None
# if "last_improved_edit" not in st.session_state:
#     st.session_state["last_improved_edit"] = None

# if run:
#     # Validate inputs
#     if not resume_text.strip():
#         st.error("Please upload a resume PDF or paste resume text.")
#     elif not target_role.strip():
#         st.error("Please enter a target job role.")
#     else:
#         with st.spinner("Calling LLM and analyzing resume..."):
#             # Build user prompt
#             prompt = build_user_prompt(resume_text, target_role.strip(), jd_text or None)
#             raw_llm_output = ""
#             # Call LLM if configured, otherwise leave raw empty (we will inject demo fallback)
#             if _openai_client:
#                 try:
#                     raw_llm_output = call_llm_api(model_choice, LLM_SYSTEM_PROMPT, prompt)
#                 except Exception as e:
#                     st.warning(f"LLM call failed: {e}. Falling back to demo output.")
#                     raw_llm_output = ""
#             else:
#                 raw_llm_output = ""

#             st.session_state["last_raw"] = raw_llm_output

#             # Parse JSON from LLM output robustly
#             parsed = extract_json_from_raw(raw_llm_output) if raw_llm_output else None

#             # If parsing failed or no LLM, provide a demo fallback structured response
#             if not parsed:
#                 # Demo fallback (useful for testing UI without OpenAI configured)
#                 parsed = {
#                     "overall_summary": "Demo fallback: resume needs better tailoring to the target role.",
#                     "keyword_gaps": ["ExampleSkillA", "ExampleSkillB"],
#                     "duplicate_or_vague_phrases": ["handled tasks", "various things"],
#                     "suggestions": [
#                         "Add measurable achievements for each role.",
#                         "Include role-specific keywords from the JD.",
#                         "Use action verbs and quantify impact."
#                     ],
#                     "section_feedback": {
#                         "Education": ["Include relevant coursework."],
#                         "Experience": ["Add metrics for accomplishments."],
#                         "Skills": ["Prioritize domain-specific skills."],
#                         "Projects": [],
#                         "Certifications": [],
#                         "SummaryOrObjective": ["Add a concise objective."],
#                         "Other": []
#                     },
#                     "scoring": {"relevance": 5, "clarity": 6, "impact": 4, "ats": 5},
#                     "improved_resume": ""  # empty so we synthesize below
#                 }

#             # Normalize scoring: may be int, dict with missing keys, etc.
#             scoring = parsed.get("scoring", {})
#             if isinstance(scoring, int):
#                 parsed["scoring"] = {
#                     "relevance": int(scoring),
#                     "clarity": 7,
#                     "impact": 7,
#                     "ats": 7
#                 }
#             elif not isinstance(scoring, dict):
#                 parsed["scoring"] = {"relevance": 0, "clarity": 0, "impact": 0, "ats": 0}
#             else:
#                 for k in ["relevance", "clarity", "impact", "ats"]:
#                     v = scoring.get(k)
#                     if not isinstance(v, int):
#                         scoring[k] = 7
#                 parsed["scoring"] = scoring

#             # Ensure section_feedback lists are lists
#             sf = parsed.get("section_feedback", {}) or {}
#             for key in ["Education", "Experience", "Skills", "Projects", "Certifications", "SummaryOrObjective", "Other"]:
#                 val = sf.get(key, [])
#                 if isinstance(val, str):
#                     sf[key] = [val]
#                 elif not isinstance(val, list):
#                     sf[key] = []
#             parsed["section_feedback"] = sf

#             # Normalize suggestions
#             suggestions = parsed.get("suggestions", [])
#             if isinstance(suggestions, str):
#                 # Split on newlines or bullets
#                 suggestions_list = [s.strip(" -•") for s in suggestions.splitlines() if s.strip()]
#                 parsed["suggestions"] = suggestions_list

#             # Ensure keyword_gaps and duplicate list types
#             for arr_key in ["keyword_gaps", "duplicate_or_vague_phrases"]:
#                 val = parsed.get(arr_key, [])
#                 if isinstance(val, str):
#                     parsed[arr_key] = [v.strip() for v in re.split(r"[,\n;]+", val) if v.strip()]
#                 elif not isinstance(val, list):
#                     parsed[arr_key] = []

#             # If improved_resume absent or empty, synthesize from parsed data
#             improved_text = parsed.get("improved_resume") or ""
#             if not isinstance(improved_text, str):
#                 improved_text = ""
#             if not improved_text.strip():
#                 improved_text = synthesize_improved_resume(parsed, target_role.strip())
#             parsed["improved_resume"] = improved_text

#             # Attempt to validate parsed structure using Pydantic (friendly error handling)
#             try:
#                 review_obj = ReviewResponse(**parsed)
#             except ValidationError as ve:
#                 # If Pydantic validation fails, show helpful debugging info and try to coerce minimally
#                 st.warning("Received LLM output didn't perfectly match schema; attempting to coerce / continue.")
#                 st.exception(ve)
#                 # Try to coerce minimal parts
#                 # Ensure lists for required fields
#                 parsed.setdefault("overall_summary", parsed.get("overall_summary", ""))
#                 parsed.setdefault("keyword_gaps", parsed.get("keyword_gaps", []))
#                 parsed.setdefault("duplicate_or_vague_phrases", parsed.get("duplicate_or_vague_phrases", []))
#                 parsed.setdefault("suggestions", parsed.get("suggestions", []))
#                 parsed.setdefault("section_feedback", parsed.get("section_feedback", {}))
#                 parsed.setdefault("scoring", parsed.get("scoring", {"relevance":0,"clarity":0,"impact":0,"ats":0}))
#                 parsed.setdefault("improved_resume", parsed.get("improved_resume", ""))
#                 # final attempt
#                 review_obj = ReviewResponse(**parsed)

#             # Save review in session
#             st.session_state["last_review"] = parsed
#             st.session_state["last_improved_edit"] = parsed["improved_resume"]

#             # ----------------- Display Results -----------------
#             st.success("Review complete.")
#             # Overall summary
#             with st.expander("Overall Summary", expanded=True):
#                 st.write(review_obj.overall_summary)

#             # Scores
#             with st.expander("Scores (0–10)", expanded=False):
#                 cols = st.columns(4)
#                 cols[0].metric("Relevance", review_obj.scoring.relevance)
#                 cols[1].metric("Clarity", review_obj.scoring.clarity)
#                 cols[2].metric("Impact", review_obj.scoring.impact)
#                 cols[3].metric("ATS", review_obj.scoring.ats)

#             # Keyword gaps
#             with st.expander("Keyword Gaps", expanded=False):
#                 if review_obj.keyword_gaps:
#                     for k in review_obj.keyword_gaps:
#                         st.markdown(f"- {k}")
#                 else:
#                     st.info("No major keyword gaps detected.")

#             # Duplicate / vague phrases
#             with st.expander("Duplicate / Vague Phrases", expanded=False):
#                 if review_obj.duplicate_or_vague_phrases:
#                     for p in review_obj.duplicate_or_vague_phrases:
#                         st.markdown(f"- {p}")
#                 else:
#                     st.write("None detected.")

#             # Section-wise feedback
#             with st.expander("Section-wise Feedback", expanded=True):
#                 sf_obj = review_obj.section_feedback
#                 # custom ordering & display
#                 mapping = [
#                     ("Summary / Objective", sf_obj.SummaryOrObjective),
#                     ("Experience", sf_obj.Experience),
#                     ("Education", sf_obj.Education),
#                     ("Skills", sf_obj.Skills),
#                     ("Projects", sf_obj.Projects),
#                     ("Certifications", sf_obj.Certifications),
#                     ("Other", sf_obj.Other),
#                 ]
#                 for label, items in mapping:
#                     if items:
#                         st.markdown(f"**{label}**")
#                         for it in items:
#                             st.markdown(f"- {it}")

#             # General suggestions (each on its own bullet)
#             with st.expander("General Suggestions", expanded=False):
#                 if review_obj.suggestions:
#                     for s in review_obj.suggestions:
#                         st.markdown(f"- {s}")
#                 else:
#                     st.write("No suggestions available.")

#             # Improved resume display + PDF download
#             if gen_improved_toggle:
#                 st.divider()
#                 st.subheader("Improved Resume (editable)")
#                 # Editable text area (store edit in session)
#                 initial_improved = st.session_state.get("last_improved_edit", review_obj.improved_resume)
#                 edited = st.text_area("Edit improved resume (you can tweak before export)", value=initial_improved, height=360, key="improved_edit_area")
#                 # Update session if user edited
#                 if edited != st.session_state.get("last_improved_edit"):
#                     st.session_state["last_improved_edit"] = edited

#                 # PDF generation and download
#                 pdf_bytes = generate_pdf_bytes_safe(edited)
#                 st.download_button(
#                     "📄 Download Improved Resume (PDF)",
#                     data=pdf_bytes,
#                     file_name="improved_resume.pdf",
#                     mime="application/pdf",
#                 )

#             # Comparison: original vs improved with diff
#             if resume_text:
#                 st.divider()
#                 st.subheader("Comparison: Original ⇢ Improved")
#                 show_diff = st.checkbox("Show unified diff", value=False)
#                 col_a, col_b = st.columns(2)
#                 with col_a:
#                     st.markdown("**Original Resume**")
#                     st.text_area("Original (read-only)", value=resume_text, height=360, key="orig_display")
#                 with col_b:
#                     st.markdown("**Improved Resume**")
#                     current_improved = st.session_state.get("last_improved_edit", review_obj.improved_resume)
#                     st.text_area("Improved (editable)", value=current_improved, height=360, key="improved_display")

#                 if show_diff:
#                     st.divider()
#                     st.subheader("Unified Diff (Original → Improved)")
#                     orig_lines = resume_text.splitlines()
#                     new_lines = st.session_state.get("last_improved_edit", review_obj.improved_resume).splitlines()
#                     diff_lines = list(difflib.unified_diff(orig_lines, new_lines, lineterm=""))
#                     if diff_lines:
#                         st.code("\n".join(diff_lines))
#                     else:
#                         st.write("No differences detected.")

# # If there is a last review stored and user did not just run, show a quick panel to reopen
# if not run and st.session_state.get("last_review"):
#     st.divider()
#     st.info("A previous review exists in this session.")
#     if st.button("Open last review and comparison"):
#         parsed = st.session_state["last_review"]
#         try:
#             review_obj = ReviewResponse(**parsed)
#             st.success("Loaded previous review.")
#             with st.expander("Overall Summary (previous)", expanded=False):
#                 st.write(review_obj.overall_summary)
#             # Show improved resume from session
#             st.subheader("Improved Resume (previous)")
#             prev_improved = st.session_state.get("last_improved_edit", review_obj.improved_resume)
#             st.text_area("Improved (editable, previous)", value=prev_improved, height=360, key="prev_improved_area")
#             pdf_prev = generate_pdf_bytes_safe(prev_improved)
#             st.download_button(
#                 "📄 Download Previous Improved (PDF)",
#                 data=pdf_prev,
#                 file_name="improved_resume_previous.pdf",
#                 mime="application/pdf",
#             )
#         except Exception as e:
#             st.error("Could not load previous review.")
#             st.exception(e)

# st.divider()
# st.markdown(
#     "**Tips & Next steps**\n"
#     "- To use a real LLM, set the OPENAI_API_KEY environment variable and choose a supported model.\n"
#     "- For prettier PDF exports (with true Unicode support), you can embed a TTF font via fpdf's `add_font` and use UTF-8 text.\n"
#     "- Add history persistence (database or file) to track improvements across uploads.\n"
#     "- Consider highlighting differences inline (HTML) for richer UX.\n"
# )

