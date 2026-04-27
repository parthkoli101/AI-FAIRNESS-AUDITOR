import os
import json
import math
import uuid
import time
import textwrap
import subprocess
import tempfile
import traceback
from io import StringIO
from datetime import datetime, timezone
from functools import wraps

from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests as http_requests

try:
    from pymongo import MongoClient
    from bson import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import groq as groq_lib
    GROQ_LIB_AVAILABLE = True
except ImportError:
    GROQ_LIB_AVAILABLE = False

try:
    import anthropic as anthropic_lib
    ANTHROPIC_LIB_AVAILABLE = True
except ImportError:
    ANTHROPIC_LIB_AVAILABLE = False

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# ─── MongoDB Setup ──────────────────────────────────────────────────────────
db = None
if MONGO_AVAILABLE:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client["fairness_ai"]
        MONGO_AVAILABLE = True
    except Exception:
        MONGO_AVAILABLE = False

# ─── AI Provider Detection ──────────────────────────────────────────────────
def get_ai_provider():
    if GROQ_API_KEY and GROQ_LIB_AVAILABLE:
        return "groq"
    elif ANTHROPIC_API_KEY and ANTHROPIC_LIB_AVAILABLE:
        return "anthropic"
    return None

# ─── Auth Helpers ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─── Smart Text-to-DataFrame Parser ──────────────────────────────────────────
def parse_text_as_dataset(text, source_hint="txt"):
    """
    Rule-based + heuristic parser that converts unstructured / semi-structured
    text (TXT, PDF extracted text, HTML, XML) into a pandas DataFrame.

    Strategy (tried in order):
    1. Detect CSV-like content inside the text (comma/tab/pipe separated)
    2. Detect repeated XML/JSON-like records and flatten them
    3. Detect key:value pair blocks (one record per blank-line block)
    4. Detect whitespace-aligned tables (fixed-width columns)
    5. Build a synthetic statistical profile from named entities / keywords
       so that bias metrics can still be computed.
    """
    import re
    import io

    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, "Empty file"

    # ── Strategy 1: CSV / TSV / PSV embedded in text ──────────────────────
    for sep, name in [(",", "csv"), ("\t", "tsv"), ("|", "psv")]:
        candidate_lines = [l for l in lines if l.count(sep) >= 2]
        if len(candidate_lines) >= 5:
            try:
                joined = "\n".join(candidate_lines)
                df = pd.read_csv(io.StringIO(joined), sep=sep, engine="python",
                                 on_bad_lines="skip")
                if len(df) >= 3 and len(df.columns) >= 2:
                    return df, None
            except Exception:
                pass

    # ── Strategy 2: XML/HTML tag extraction ──────────────────────────────
    # Find all unique tag names, try to build a record per repeating root
    tag_pattern = re.compile(r"<([A-Za-z_][A-Za-z0-9_\-]*)(?:\s[^>]*)?>([^<]*)</\1>")
    matches = tag_pattern.findall(text)
    if len(matches) >= 10:
        # Group by discovering repeating blocks
        tag_names = [m[0].lower() for m in matches]
        tag_counts = {}
        for t in tag_names:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        # Use the most common tag as record separator
        common_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
        # Build rows: find enclosing record tags
        record_tag_candidates = [t for t, c in common_tags if c >= 3]
        # Try to find a parent/record tag that contains children
        record_re = re.compile(
            r"<([A-Za-z_][A-Za-z0-9_\-]*)(?:\s[^>]*)?>(.+?)</\1>",
            re.DOTALL | re.IGNORECASE
        )
        # Build flat records from all tag:value pairs grouped by position
        all_matches = [(m[0].lower(), m[1].strip()) for m in tag_pattern.findall(text)]
        if all_matches:
            # Detect field names = tags that appear roughly equally
            from collections import Counter
            field_counts = Counter(t for t, v in all_matches)
            n_records = max(field_counts.values())
            fields = [t for t, c in field_counts.items() if c >= max(1, n_records // 2)]
            if len(fields) >= 2:
                # Build DataFrame row by row
                rows = []
                current = {}
                seen_fields = set()
                for tag, val in all_matches:
                    if tag in fields:
                        if tag in seen_fields:
                            # New record
                            rows.append({f: current.get(f, "") for f in fields})
                            current = {}
                            seen_fields = set()
                        current[tag] = val
                        seen_fields.add(tag)
                if current:
                    rows.append({f: current.get(f, "") for f in fields})
                if len(rows) >= 3:
                    df = pd.DataFrame(rows)
                    return df, None

    # ── Strategy 3: Key:Value block records ───────────────────────────────
    kv_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_ \-/]*)[\s]*[:=]\s*(.+)$")
    blocks = []
    current_block = {}
    for line in lines:
        m = kv_pattern.match(line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip()
            current_block[key] = val
        else:
            if current_block:
                blocks.append(current_block)
                current_block = {}
    if current_block:
        blocks.append(current_block)

    if len(blocks) >= 3:
        # Find fields present in at least half the blocks
        from collections import Counter
        all_keys = Counter(k for b in blocks for k in b.keys())
        n = len(blocks)
        common_keys = [k for k, c in all_keys.items() if c >= max(2, n // 3)]
        if len(common_keys) >= 2:
            rows = [{k: b.get(k, "") for k in common_keys} for b in blocks]
            df = pd.DataFrame(rows)
            return df, None

    # ── Strategy 4: Fixed-width / whitespace table ────────────────────────
    if len(lines) >= 4:
        # Find candidate header line: all alpha tokens separated by 2+ spaces
        header_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*(\s{2,}[A-Za-z_][A-Za-z0-9_ ]*){2,}$")
        header_idx = None
        for i, line in enumerate(lines[:20]):
            if header_re.match(line.strip()):
                header_idx = i
                break
        if header_idx is not None:
            try:
                table_text = "\n".join(lines[header_idx:])
                df = pd.read_fwf(io.StringIO(table_text))
                if len(df) >= 3 and len(df.columns) >= 2:
                    return df, None
            except Exception:
                pass

    # ── Strategy 5: Keyword / entity frequency synthetic profile ─────────
    # For truly unstructured text (essays, reports, narratives), we extract
    # demographic mentions and build a synthetic frequency table that can
    # be used for representation analysis.
    demographic_patterns = {
        "gender": re.compile(r"\b(male|female|man|woman|men|women|non.binary|nonbinary|transgender|gender)\b", re.I),
        "race": re.compile(r"\b(white|black|hispanic|latino|latina|asian|african|caucasian|indigenous|native|race|racial|ethnic)\b", re.I),
        "age": re.compile(r"\b(\d{2})\s*(?:year|yr)s?\s*old|\bage\s+(\d{2})|\b(young|old|senior|junior|elderly|youth|adult|teenager)\b", re.I),
        "disability": re.compile(r"\b(disabled|disability|handicap|impair|wheelchair|blind|deaf|autis|adhd)\b", re.I),
        "religion": re.compile(r"\b(christian|muslim|jewish|hindu|buddhist|religious|faith|church|mosque|temple)\b", re.I),
        "nationality": re.compile(r"\b(american|british|indian|chinese|mexican|canadian|nationality|immigrant|foreign|citizen)\b", re.I),
    }

    outcome_patterns = {
        "hired": re.compile(r"\b(hired|hire|accepted|approved|selected|admitted|passed|qualified)\b", re.I),
        "rejected": re.compile(r"\b(rejected|denied|refused|failed|disqualified|not selected|not hired)\b", re.I),
    }

    # Count occurrences per paragraph/sentence to build synthetic rows
    sentences = re.split(r"[.!?\n]+", text)
    rows = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        row = {}
        for attr, pat in demographic_patterns.items():
            matches_found = pat.findall(sent)
            if matches_found:
                val = matches_found[0]
                if isinstance(val, tuple):
                    val = next((v for v in val if v), attr)
                row[attr] = str(val).lower().strip()
        for outcome, pat in outcome_patterns.items():
            if pat.search(sent):
                row["outcome"] = outcome
                break
        if row and len(row) >= 2:
            rows.append(row)

    if len(rows) >= 5:
        df = pd.DataFrame(rows)
        # Fill missing with "unknown"
        df = df.fillna("unknown")
        return df, None

    # ── Fallback: frequency count of demographic terms as distribution ────
    term_counts = {}
    for attr, pat in demographic_patterns.items():
        found = pat.findall(text)
        if found:
            for match in found:
                if isinstance(match, tuple):
                    match = next((v for v in match if v), attr)
                key = f"{attr}_{str(match).lower().strip()}"
                term_counts[key] = term_counts.get(key, 0) + 1

    if term_counts:
        rows = [{"term": k, "frequency": v, "category": k.split("_")[0]}
                for k, v in term_counts.items()]
        df = pd.DataFrame(rows)
        return df, None

    return None, "Could not extract structured data from this file"


# ─── File Loader ─────────────────────────────────────────────────────────────
def load_uploaded_file(file):
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        try:
            df = pd.read_csv(file)
            return "dataframe", df
        except Exception as e:
            return "error", str(e)

    elif ext == "json":
        try:
            raw = json.load(file)
            if isinstance(raw, list):
                df = pd.json_normalize(raw)
            elif isinstance(raw, dict):
                df = pd.json_normalize(raw)
            else:
                return "error", "Unsupported JSON structure"
            return "dataframe", df
        except Exception as e:
            return "error", str(e)

    elif ext in ("xlsx", "xls"):
        try:
            df = pd.read_excel(file)
            return "dataframe", df
        except Exception as e:
            return "error", str(e)

    elif ext == "xml":
        try:
            import xml.etree.ElementTree as ET
            raw_bytes = file.read()
            text = raw_bytes.decode("utf-8", errors="replace")
            # First try: parse as proper XML and flatten records
            try:
                root = ET.fromstring(raw_bytes)
                # Find the most repeated child tag (record tag)
                from collections import Counter
                child_tags = Counter(child.tag for child in root)
                if child_tags:
                    record_tag = child_tags.most_common(1)[0][0]
                    rows = []
                    for record in root.findall(record_tag):
                        row = {}
                        # direct children as fields
                        for field in record:
                            row[field.tag.lower()] = field.text or ""
                        # attributes as fields
                        for attr_name, attr_val in record.attrib.items():
                            row[attr_name.lower()] = attr_val
                        if row:
                            rows.append(row)
                    if len(rows) >= 2:
                        return "dataframe", pd.DataFrame(rows)
                # Try root itself as a single record with nested children
                rows = []
                for child in root:
                    row = {"tag": child.tag.lower()}
                    row["value"] = child.text or ""
                    for sub in child:
                        row[sub.tag.lower()] = sub.text or ""
                    for attr_name, attr_val in child.attrib.items():
                        row[attr_name.lower()] = attr_val
                    rows.append(row)
                if rows:
                    return "dataframe", pd.DataFrame(rows)
            except ET.ParseError:
                pass
            # Fallback: treat as text and use smart parser
            df, err = parse_text_as_dataset(text, "xml")
            if df is not None and len(df) >= 3:
                return "dataframe", df
            return "text", text
        except Exception as e:
            return "error", str(e)

    elif ext in ("html", "htm"):
        try:
            raw_bytes = file.read()
            text = raw_bytes.decode("utf-8", errors="replace")
            # Strategy 1: pandas read_html for <table> tags
            import io
            try:
                tables = pd.read_html(io.StringIO(text))
                if tables:
                    # Pick the largest table
                    best = max(tables, key=lambda t: len(t))
                    if len(best) >= 2 and len(best.columns) >= 2:
                        return "dataframe", best
            except Exception:
                pass
            # Strategy 2: strip HTML tags and use smart text parser
            import re
            clean = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.I)
            clean = re.sub(r"<style[^>]*>.*?</style>", " ", clean, flags=re.DOTALL | re.I)
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            df, err = parse_text_as_dataset(clean, "html")
            if df is not None and len(df) >= 3:
                return "dataframe", df
            return "text", clean
        except Exception as e:
            return "error", str(e)

    elif ext == "txt":
        try:
            text = file.read().decode("utf-8", errors="replace")
            # Smart parse: try to extract structured data first
            df, err = parse_text_as_dataset(text, "txt")
            if df is not None and len(df) >= 3:
                return "dataframe", df
            # Fallback to text mode for AI content audit
            return "text", text
        except Exception as e:
            return "error", str(e)

    elif ext == "pdf":
        try:
            if PDF_AVAILABLE:
                with pdfplumber.open(file) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(pages)
            else:
                text = file.read().decode("utf-8", errors="replace")
            # Smart parse: try to extract structured data first
            df, err = parse_text_as_dataset(text, "pdf")
            if df is not None and len(df) >= 3:
                return "dataframe", df
            return "text", text
        except Exception as e:
            return "error", str(e)

    else:
        try:
            text = file.read().decode("utf-8", errors="replace")
            df, err = parse_text_as_dataset(text, ext)
            if df is not None and len(df) >= 3:
                return "dataframe", df
            return "text", text
        except Exception:
            return "error", f"Unsupported file type: {ext}"

# ─── MongoDB Normalizer ───────────────────────────────────────────────────────
def normalize_for_mongo(obj):
    if isinstance(obj, dict):
        return {str(k): normalize_for_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_for_mongo(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

# ─── Protected Attribute Detection ───────────────────────────────────────────
PROTECTED_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "disability", "religion",
    "nationality", "age", "marital", "pregnancy", "origin"
]

def detect_protected_attributes(df):
    found = []
    for col in df.columns:
        col_lower = col.lower()
        for kw in PROTECTED_KEYWORDS:
            if kw in col_lower:
                found.append(col)
                break
    return found

def bin_age_column(df):
    df = df.copy()
    for col in df.columns:
        if "age" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            bins = [0, 25, 35, 45, 60, 200]
            labels = ["18-25", "26-35", "36-45", "46-60", "60+"]
            df[col] = pd.cut(df[col], bins=bins, labels=labels, right=True)
    return df

TARGET_KEYWORDS = [
    "hired", "approved", "selected", "accepted", "rejected",
    "loan_approved", "admitted", "decision", "outcome",
    "prediction", "result", "label"
]

POSITIVE_VALUES = {
    "1", "yes", "true", "approved", "accept", "accepted",
    "selected", "hired", "positive", "pass"
}

def detect_target_column(df):
    for col in df.columns:
        if col.lower().strip() in TARGET_KEYWORDS:
            return col
    return None

def convert_to_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).apply(lambda x: 1 if x > 0 else 0)
    return series.astype(str).str.lower().str.strip().apply(
        lambda x: 1 if x in POSITIVE_VALUES else 0
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIX #2: PROPER NORMALIZED BIAS SCORE
# Uses bounded SPD + DI metrics. Never outputs 100 unless complete exclusion.
# Thresholds: 0-15 = Fair, 15-35 = Medium, 35-60 = High, 60+ = Critical
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bias_score(metrics):
    """
    Normalized bias score using SPD and DI. Output range: 0-100.
    Only outputs 100 if a group has a 0% selection rate (complete exclusion).

    Scoring logic:
    - SPD (Statistical Parity Difference): 0 = fair, 1 = max bias
    - DI (Disparate Impact): 1.0 = fair, 0 = max bias -> bias_from_di = 1 - DI
    - Take worst-case across all protected attributes
    - Apply piecewise scaling to meaningful thresholds
    """
    spd_dict = metrics.get("statistical_parity_difference", {})
    di_dict = metrics.get("disparate_impact", {})

    if isinstance(spd_dict, (int, float)):
        spd_dict = {"_": spd_dict}
    if isinstance(di_dict, (int, float)):
        di_dict = {"_": di_dict}

    spd_scores, di_scores = [], []

    for spd in spd_dict.values():
        if spd is not None:
            # SPD is already in 0-1 range (difference of rates)
            spd_scores.append(min(max(float(spd), 0.0), 1.0))
    for di in di_dict.values():
        if di is not None:
            # DI: 1.0 = fair, 0 = max bias. Clamp to [0, 1]
            di_val = min(max(float(di), 0.0), 1.0)
            # bias_from_di = 1 - DI, but only meaningful if DI < 1
            di_bias = 1.0 - di_val
            di_scores.append(di_bias)

    worst_spd = max(spd_scores) if spd_scores else 0.0
    worst_di = max(di_scores) if di_scores else 0.0

    # Use the worse of the two metrics
    combined = max(worst_spd, worst_di)

    # Piecewise scaling - maps metric range [0,1] to intuitive score [0,100]
    # designed so that moderate disparities don't hit 100
    if combined == 0:
        base_score = 0.0
    elif combined <= 0.05:
        # 0-5% disparity -> score 0-7 (essentially fair)
        base_score = combined * 140
    elif combined <= 0.10:
        # 5-10% -> score 7-15 (low bias range)
        base_score = 7 + (combined - 0.05) * 160
    elif combined <= 0.20:
        # 10-20% -> score 15-28 (low-medium)
        base_score = 15 + (combined - 0.10) * 130
    elif combined <= 0.35:
        # 20-35% -> score 28-45 (medium)
        base_score = 28 + (combined - 0.20) * 113
    elif combined <= 0.50:
        # 35-50% -> score 45-60 (high)
        base_score = 45 + (combined - 0.35) * 100
    elif combined <= 0.70:
        # 50-70% -> score 60-78 (high-critical)
        base_score = 60 + (combined - 0.50) * 90
    elif combined <= 0.90:
        # 70-90% -> score 78-92 (critical)
        base_score = 78 + (combined - 0.70) * 70
    else:
        # 90-100% -> score 92-99 (near complete exclusion)
        base_score = 92 + (combined - 0.90) * 70

    # Small penalties for data quality issues (capped to avoid inflating score)
    missing_pct = metrics.get("missing_pct", {})
    missing_penalty = min(sum(1 for v in missing_pct.values() if v > 20) * 2, 5)
    alert_penalty = min(metrics.get("alert_count", 0) * 2, 5)

    final_score = base_score + missing_penalty + alert_penalty

    # Check for complete exclusion (selection_rate = 0 for any group)
    complete_exclusion = False
    for group_rates in list(metrics.get("outcome_bias", {}).values()) + list(metrics.get("group_rates", {}).values()):
        if isinstance(group_rates, dict):
            for v in group_rates.values():
                if v == 0.0:
                    complete_exclusion = True
                    break

    # NEVER output 100 unless complete exclusion confirmed
    if not complete_exclusion:
        final_score = min(final_score, 97.0)

    return round(min(final_score, 100.0), 1)


def score_to_risk(score):
    if score < 15:
        return "LOW"
    elif score < 35:
        return "MEDIUM"
    elif score < 60:
        return "HIGH"
    return "CRITICAL"


def compute_audit_confidence(metrics):
    score = 100
    if not metrics.get("protected_attributes"):
        score -= 30
    if not metrics.get("distributions") and not metrics.get("group_rates"):
        score -= 25
    if not metrics.get("total_rows") or metrics.get("total_rows", 0) < 100:
        score -= 20
    if score >= 80:
        return "HIGH"
    elif score >= 55:
        return "MEDIUM"
    return "LOW"


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT / PDF / XML / HTML METRICS EXTRACTOR
# Pure rule-based NLP that converts unstructured text into fairness metrics.
# Works WITHOUT an LLM — counts demographic mentions, outcome associations,
# and representation gaps so the full dashboard pipeline can run normally.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_text_metrics(text, filename="document"):
    """
    Extract demographic distributions and outcome bias from raw text.
    Returns a metrics dict compatible with compute_pretraining_metrics output
    so graphs, alerts and reports all work the same as for CSV files.
    """
    import re
    from collections import Counter

    # ── 1. Demographic group patterns ──────────────────────────────────────
    DEMO_PATTERNS = {
        "gender": {
            "male":       re.compile(r"\b(male|man|men|boy|his|he)\b", re.I),
            "female":     re.compile(r"\b(female|woman|women|girl|her|she)\b", re.I),
            "non-binary": re.compile(r"\b(non.binary|nonbinary|they|them|enby|genderqueer)\b", re.I),
        },
        "race": {
            "white":     re.compile(r"\b(white|caucasian|european)\b", re.I),
            "black":     re.compile(r"\b(black|african.american|afro)\b", re.I),
            "hispanic":  re.compile(r"\b(hispanic|latino|latina|latinx)\b", re.I),
            "asian":     re.compile(r"\b(asian|chinese|japanese|korean|indian)\b", re.I),
            "indigenous":re.compile(r"\b(indigenous|native|aboriginal|first.nation)\b", re.I),
        },
        "age_group": {
            "18-25":  re.compile(r"\b(1[89]|2[0-5])\s*(?:year|yr)s?\s*old|\byouth\b|\bteenager\b|\byoung adult\b", re.I),
            "26-35":  re.compile(r"\b(2[6-9]|3[0-5])\s*(?:year|yr)s?\s*old|\byoung professional\b", re.I),
            "36-45":  re.compile(r"\b(3[6-9]|4[0-5])\s*(?:year|yr)s?\s*old|\bmid.career\b", re.I),
            "46-60":  re.compile(r"\b(4[6-9]|5[0-9]|60)\s*(?:year|yr)s?\s*old|\bsenior employee\b", re.I),
            "60+":    re.compile(r"\b(6[1-9]|[789]\d)\s*(?:year|yr)s?\s*old|\belderly\b|\bretir", re.I),
        },
        "disability": {
            "disabled":      re.compile(r"\b(disab|handicap|impair|wheelchair|blind|deaf|autis|adhd|dyslexia)\w*", re.I),
            "non-disabled":  re.compile(r"\b(able.bodied|neurotypical|no disability|without disab)\b", re.I),
        },
        "religion": {
            "christian": re.compile(r"\b(christian|church|catholic|protestant|baptist|evangelical)\b", re.I),
            "muslim":    re.compile(r"\b(muslim|islam|mosque|hijab|halal)\b", re.I),
            "jewish":    re.compile(r"\b(jewish|jew|synagogue|kosher|rabbi)\b", re.I),
            "hindu":     re.compile(r"\b(hindu|temple|diwali|brahmin)\b", re.I),
            "other":     re.compile(r"\b(buddhist|sikh|atheist|agnostic|secular)\b", re.I),
        },
        "nationality": {
            "domestic":      re.compile(r"\b(citizen|national|domestic|born here|local)\b", re.I),
            "international": re.compile(r"\b(immigrant|foreign|overseas|visa|migrant|expat)\b", re.I),
        },
    }

    OUTCOME_PATTERNS = {
        "positive": re.compile(
            r"\b(hired|accepted|approved|selected|admitted|passed|qualified|promoted|"
            r"successful|granted|offered|recruited|shortlisted|advanced)\b", re.I),
        "negative": re.compile(
            r"\b(rejected|denied|refused|failed|disqualified|dismissed|terminated|"
            r"not selected|not hired|not approved|turned down|laid off)\b", re.I),
    }

    # ── 2. Count raw mentions ───────────────────────────────────────────────
    group_counts = {}
    for attr, groups in DEMO_PATTERNS.items():
        counts = {}
        for group_name, pat in groups.items():
            c = len(pat.findall(text))
            if c > 0:
                counts[group_name] = c
        if counts:
            group_counts[attr] = counts

    total_mentions = sum(sum(g.values()) for g in group_counts.values())
    total_chars = len(text)
    total_words = len(text.split())

    # ── 3. Sentence-level co-occurrence for outcome bias ────────────────────
    sentences = re.split(r"[.!?\n]+", text)
    outcome_by_group = {}   # attr -> group -> {pos: int, neg: int, total: int}

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        pos = bool(OUTCOME_PATTERNS["positive"].search(sent))
        neg = bool(OUTCOME_PATTERNS["negative"].search(sent))
        if not pos and not neg:
            continue
        for attr, groups in DEMO_PATTERNS.items():
            for group_name, pat in groups.items():
                if pat.search(sent):
                    if attr not in outcome_by_group:
                        outcome_by_group[attr] = {}
                    if group_name not in outcome_by_group[attr]:
                        outcome_by_group[attr][group_name] = {"pos": 0, "neg": 0, "total": 0}
                    if pos:
                        outcome_by_group[attr][group_name]["pos"] += 1
                    if neg:
                        outcome_by_group[attr][group_name]["neg"] += 1
                    outcome_by_group[attr][group_name]["total"] += 1

    # ── 4. Build metrics dict ───────────────────────────────────────────────
    metrics = {}
    metrics["total_rows"] = total_words          # proxy for dataset "size"
    metrics["total_columns"] = len(group_counts) # each attr = a "column"
    metrics["columns"] = list(group_counts.keys())
    metrics["source"] = "text_extraction"
    metrics["file_type"] = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    metrics["text_stats"] = {
        "total_characters": total_chars,
        "total_words": total_words,
        "total_sentences": len([s for s in sentences if len(s.strip()) > 10]),
        "demographic_mentions": total_mentions,
    }

    # Protected attributes = any attr we found mentions for
    metrics["protected_attributes"] = list(group_counts.keys())

    # Distributions (normalised counts = proxy for representation)
    distributions = {}
    for attr, counts in group_counts.items():
        total = sum(counts.values())
        if total > 0:
            distributions[attr] = {g: round(c / total, 4) for g, c in counts.items()}
    metrics["distributions"] = distributions

    # Missing: no concept for text files
    metrics["missing_values"] = {}
    metrics["missing_pct"] = {}

    # Outcome bias from co-occurrence
    outcome_bias = {}
    spd_values = {}
    di_values = {}
    alerts = []
    target_col = "document_outcome"

    for attr, groups in outcome_by_group.items():
        # Only include groups that have both pos and neg signal
        group_rates = {}
        for group_name, counts in groups.items():
            total = counts["total"]
            if total > 0:
                rate = round(counts["pos"] / total, 4)
                group_rates[group_name] = rate

        if len(group_rates) >= 2:
            rates = list(group_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            spd = round(max_rate - min_rate, 4)
            di = round(min_rate / max_rate, 4) if max_rate > 0 else 0
            outcome_bias[attr] = group_rates
            spd_values[attr] = spd
            di_values[attr] = di
            if spd > 0.10 or di < 0.80:
                alerts.append({
                    "type": "outcome_bias",
                    "column": attr,
                    "target": target_col,
                    "statistical_parity_difference": spd,
                    "disparate_impact": di,
                    "detail": f"Outcome bias in '{attr}': SPD={spd}, DI={di}",
                    "severity": "HIGH" if (spd > 0.20 or di < 0.60) else "MEDIUM"
                })

    # Representation imbalance alerts
    for attr, dist in distributions.items():
        if dist:
            max_val = max(dist.values())
            if max_val > 0.70:
                dominant = max(dist, key=dist.get)
                alerts.append({
                    "type": "imbalance",
                    "column": attr,
                    "dominant_group": dominant,
                    "pct": round(max_val * 100, 1),
                    "severity": "HIGH" if max_val > 0.85 else "MEDIUM"
                })

    metrics["outcome_bias"] = outcome_bias
    metrics["statistical_parity_difference"] = spd_values
    metrics["disparate_impact"] = di_values
    metrics["target_column"] = target_col if outcome_bias else None
    metrics["alerts"] = alerts
    metrics["alert_count"] = len(alerts)

    # Bias score
    metrics["bias_score"] = compute_bias_score(metrics)
    metrics["risk_level"] = score_to_risk(metrics["bias_score"])
    metrics["audit_confidence"] = "MEDIUM" if total_mentions >= 20 else "LOW"

    return metrics


def generate_text_graph_data(metrics):
    """Generate graph data from text-extracted metrics (same format as generate_graph_data)."""
    graphs = []

    # Distribution graphs
    for attr, dist in metrics.get("distributions", {}).items():
        if dist:
            graphs.append({
                "type": "bar",
                "title": f"{attr.replace('_', ' ').title()} Mentions",
                "labels": list(dist.keys()),
                "data": [round(v * 100, 1) for v in dist.values()],
                "attr": attr
            })

    # Outcome bias graphs
    for attr, group_rates in metrics.get("outcome_bias", {}).items():
        if group_rates:
            graphs.append({
                "type": "bar",
                "title": f"Positive Outcome Rate by {attr.replace('_', ' ').title()}",
                "labels": list(group_rates.keys()),
                "data": [round(v * 100, 1) for v in group_rates.values()],
                "attr": attr
            })

    # Text stats
    ts = metrics.get("text_stats", {})
    if ts:
        graphs.append({
            "type": "bar",
            "title": "Document Overview",
            "labels": ["Words", "Sentences", "Demo Mentions"],
            "data": [
                ts.get("total_words", 0),
                ts.get("total_sentences", 0),
                ts.get("demographic_mentions", 0)
            ],
            "attr": "text_stats"
        })

    return graphs


# ═══════════════════════════════════════════════════════════════════════════════
# FIX #1 + #3 COMBINED: GROQ-POWERED ALERT EXPLANATIONS
# Python computes ALL facts. Groq writes rich natural language explanations.
# These are stored in DB alongside graphs and reports.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_alert_explanation_groq(alert, target_col=None):
    """
    Uses Groq/Anthropic to write a clear, rich plain-English explanation for a bias alert.
    Falls back to a pure-Python explanation if no AI provider is available.
    This is the MAIN bias explanation shown in the UI - must be high quality.
    """
    provider = get_ai_provider()
    if not provider:
        return _fallback_alert_explanation(alert)

    attr = alert.get("column", "unknown")
    alert_type = alert.get("type", "")
    severity = alert.get("severity", "HIGH")

    if alert_type == "outcome_bias":
        spd = alert.get("statistical_parity_difference", 0)
        di = alert.get("disparate_impact", 0)
        target = target_col or alert.get("target", "the outcome")
        spd_pct = round(float(spd) * 100, 1)
        di_pct = round(float(di) * 100, 1)
        rule80_status = "FAILS the 80% rule" if float(di) < 0.80 else "passes the 80% rule"

        prompt = f"""You are an AI fairness expert writing a bias alert explanation for a hiring/selection AI system dashboard.

DETECTED BIAS FACTS (use these exact numbers, do not change them):
- Protected attribute: {attr}
- Affected outcome column: {target}
- Statistical Parity Difference (SPD): {spd} — one group is {spd_pct}% more likely to receive a positive outcome
- Disparate Impact ratio (DI): {di} — the disadvantaged group is only {di_pct}% as likely to be selected as the dominant group
- 80% Rule status: {rule80_status}
- Severity: {severity}

Write a bias explanation with EXACTLY these 3 parts, in plain text, no markdown:

WHERE BIAS EXISTS:
[One clear sentence: state exactly which attribute ('{attr}') shows bias affecting which outcome ('{target}'). Name the disparity direction.]

WHY BIAS EXISTS:
[Two sentences: explain what the SPD and DI values mean in plain terms for a hiring context. Use the exact numbers. Explain what real-world impact this has on the disadvantaged group.]

WHAT TO DO:
[One sentence: one specific, actionable recommendation to reduce this bias.]

Rules: Use only the facts above. No invented numbers. No legal references. No markdown. Plain text only."""

    elif alert_type == "imbalance":
        pct = alert.get("pct", 0)
        dominant = alert.get("dominant_group", "unknown")

        prompt = f"""You are an AI fairness expert writing a dataset imbalance alert for a hiring AI dashboard.

DETECTED IMBALANCE FACTS (use these exact numbers):
- Protected attribute: {attr}
- Dominant group: {dominant} — makes up {pct}% of the dataset
- All other groups share only {round(100 - float(pct), 1)}% of the dataset combined
- Severity: {severity}

Write an imbalance explanation with EXACTLY these 3 parts, in plain text, no markdown:

WHERE BIAS EXISTS:
[One sentence: state that '{attr}' is severely imbalanced — '{dominant}' dominates at {pct}%.]

WHY BIAS EXISTS:
[Two sentences: explain how this imbalance causes a trained AI model to develop biased behavior. State that underrepresented groups will be systematically disadvantaged in model decisions.]

WHAT TO DO:
[One sentence: specific recommendation — oversample minority groups, collect more data, or apply class weights during training.]

Rules: Use only the facts above. No invented numbers. Plain text only. No markdown."""
    else:
        return _fallback_alert_explanation(alert)

    try:
        if provider == "groq":
            client = groq_lib.Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        elif provider == "anthropic":
            client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
    except Exception:
        traceback.print_exc()

    return _fallback_alert_explanation(alert)


def _fallback_alert_explanation(alert):
    attr = alert.get("column", "unknown")
    alert_type = alert.get("type", "")
    if alert_type == "outcome_bias":
        spd = alert.get("statistical_parity_difference", 0)
        di = alert.get("disparate_impact", 0)
        target = alert.get("target", "the outcome")
        spd_pct = round(float(spd) * 100, 1)
        di_pct = round(float(di) * 100, 1)
        return (
            f"WHERE BIAS EXISTS:\n"
            f"Outcome bias detected in '{target}' for the protected attribute '{attr}'.\n\n"
            f"WHY BIAS EXISTS:\n"
            f"One group is {spd_pct}% more likely to receive a positive outcome (SPD={spd}). "
            f"The disadvantaged group is only {di_pct}% as likely to be selected as the dominant group (DI={di}), "
            f"which {'fails' if float(di) < 0.80 else 'passes'} the 80% fairness rule.\n\n"
            f"WHAT TO DO:\n"
            f"Apply fairness constraints during model training and audit decision thresholds per group."
        )
    elif alert_type == "imbalance":
        pct = alert.get("pct", 0)
        dominant = alert.get("dominant_group", "unknown")
        return (
            f"WHERE BIAS EXISTS:\n"
            f"Severe representation imbalance in '{attr}': '{dominant}' makes up {pct}% of the dataset.\n\n"
            f"WHY BIAS EXISTS:\n"
            f"With {pct}% of data belonging to '{dominant}', the AI model will learn patterns biased toward this group. "
            f"Underrepresented groups account for only {round(100 - float(pct), 1)}% of training data, causing systematic disadvantage.\n\n"
            f"WHAT TO DO:\n"
            f"Oversample underrepresented groups or apply class weighting to balance the training distribution."
        )
    return (
        f"WHERE BIAS EXISTS:\nBias alert detected in column '{attr}'.\n\n"
        f"WHY BIAS EXISTS:\nThis attribute shows potential discriminatory patterns requiring review.\n\n"
        f"WHAT TO DO:\nConduct a detailed audit of this attribute's distribution and impact on outcomes."
    )


# ─── Graph Analysis ─────────────────────────────────────────────────────────

def compute_graph_analysis(graph):
    labels = graph.get("labels", [])
    values = graph.get("data", graph.get("values", []))
    title = graph.get("title", "")

    if not labels or not values or len(labels) != len(values):
        return None

    safe_vals, safe_labels = [], []
    for l, v in zip(labels, values):
        try:
            safe_vals.append(float(v))
            safe_labels.append(str(l))
        except (TypeError, ValueError):
            pass

    if len(safe_vals) < 2:
        return None

    max_val = max(safe_vals)
    min_val = min(safe_vals)
    max_idx = safe_vals.index(max_val)
    min_idx = safe_vals.index(min_val)
    total = sum(safe_vals)
    avg_val = total / len(safe_vals) if safe_vals else 0

    if max_val > 0:
        relative_disparity = (max_val - min_val) / max_val
    else:
        relative_disparity = 0.0

    absolute_gap = max_val - min_val
    is_distribution = total > 50
    is_rate = total <= 2.0

    if is_rate:
        risk_flag = "HIGH" if absolute_gap > 0.20 else ("MEDIUM" if absolute_gap > 0.10 else "LOW")
    elif is_distribution:
        risk_flag = "HIGH" if relative_disparity > 0.50 else ("MEDIUM" if relative_disparity > 0.30 else "LOW")
    else:
        risk_flag = "HIGH" if absolute_gap > 30 else ("MEDIUM" if absolute_gap > 15 else "LOW")

    return {
        "title": title,
        "dominant_group": safe_labels[max_idx],
        "dominant_value": round(max_val, 4),
        "lowest_group": safe_labels[min_idx],
        "lowest_value": round(min_val, 4),
        "absolute_gap": round(absolute_gap, 4),
        "relative_disparity_pct": round(relative_disparity * 100, 1),
        "average_value": round(avg_val, 4),
        "risk_flag": risk_flag,
        "is_balanced": relative_disparity < 0.15,
        "all_groups": {l: round(v, 4) for l, v in zip(safe_labels, safe_vals)},
        "group_count": len(safe_labels),
    }


def explain_graph_with_groq(graph_summary):
    analysis = compute_graph_analysis(graph_summary)
    if analysis is None:
        return "Insufficient data for analysis."

    dom = analysis["dominant_group"]
    dom_val = analysis["dominant_value"]
    low = analysis["lowest_group"]
    low_val = analysis["lowest_value"]
    gap = analysis["absolute_gap"]
    rel_pct = analysis["relative_disparity_pct"]
    risk = analysis["risk_flag"]
    title = analysis["title"]
    is_balanced = analysis["is_balanced"]
    groups_str = ", ".join([f"{g}: {v}" for g, v in analysis["all_groups"].items()])

    if is_balanced and risk == "LOW":
        risk_conclusion = f"BALANCED — no significant bias detected. The gap of {rel_pct}% is within the acceptable range."
        affected_line = "No significantly affected group — distribution is within acceptable range."
        risk_line = "Low risk — no significant fairness concern detected in this chart."
        rec_line = "Continue monitoring group distributions over time to ensure balance is maintained."
    elif risk == "MEDIUM":
        risk_conclusion = f"MODERATE disparity detected. {dom} is overrepresented. Gap of {rel_pct}% warrants monitoring."
        affected_line = f"{low} is the most affected group with a lower rate/representation."
        risk_line = f"Medium risk — {dom} has a {rel_pct}% advantage over {low}. This may affect fairness."
        rec_line = f"Monitor group rates closely and consider rebalancing {low} representation in the dataset."
    else:
        risk_conclusion = f"HIGH disparity detected. {dom} significantly dominates {low} with a {rel_pct}% gap."
        affected_line = f"{low} is the most affected group with value {low_val} vs {dom}'s {dom_val}."
        risk_line = f"High risk — {rel_pct}% disparity may lead to systematic bias against {low} in hiring/selection."
        rec_line = f"Rebalance the dataset: increase {low} representation or apply fairness constraints during retraining."

    prompt = f"""You are an AI fairness auditor writing a brief chart explanation.

PYTHON-COMPUTED FACTS — USE THESE EXACT VALUES, DO NOT CHANGE THEM:
Chart: {title}
Groups: {groups_str}
Highest group: {dom} = {dom_val}
Lowest group: {low} = {low_val}
Gap: {gap} ({rel_pct}% relative difference)
Python risk assessment: {risk}
Python conclusion: {risk_conclusion}

YOUR TASK: Write natural-language explanations using ONLY the facts above.

STRICT RULES (violation = failure):
1. Use ONLY the numbers shown above. NEVER invent or change any number.
2. If Python says LOW/BALANCED -> you MUST NOT flag it as risky. Say no significant bias.
3. If Python says HIGH -> explain disparity using the exact values above.
4. No civic, political, public-participation, or social language.
5. No elections, voting, outreach, or community programs.
6. Hiring/selection fairness context ONLY.
7. 1-2 sentences per section. Do NOT repeat points.

Return EXACTLY this format, no markdown, no extra text:

Key Finding:
[One sentence stating what the chart shows using exact values.]

Evidence:
[One sentence citing exact group values from the facts above.]

Affected Group:
[{affected_line}]

Risk:
[{risk_line}]

Recommendation:
[{rec_line}]"""

    provider = get_ai_provider()
    try:
        if provider == "groq" and GROQ_LIB_AVAILABLE:
            client = groq_lib.Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        elif provider == "anthropic" and ANTHROPIC_LIB_AVAILABLE:
            client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
    except Exception:
        traceback.print_exc()

    return f"""Key Finding:
{"The " + title + " shows a balanced distribution with no significant bias detected." if is_balanced and risk == "LOW" else f"The {title} shows a {risk.lower()} disparity — {dom} ({dom_val}) significantly exceeds {low} ({low_val})."}

Evidence:
{"Groups: " + groups_str + f". Relative gap is only {rel_pct}%, within the acceptable range." if is_balanced else f"Exact values: {groups_str}. The absolute gap is {gap} ({rel_pct}% relative difference)."}

Affected Group:
{affected_line}

Risk:
{risk_line}

Recommendation:
{rec_line}"""


# ─── Pre-Training Metrics ─────────────────────────────────────────────────────
def compute_pretraining_metrics(df):
    metrics = {}
    metrics["total_rows"] = int(len(df))
    metrics["total_columns"] = int(len(df.columns))
    metrics["columns"] = list(df.columns)

    df_binned = bin_age_column(df)
    protected_attrs = detect_protected_attributes(df_binned)
    metrics["protected_attributes"] = protected_attrs

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    metrics["missing_values"] = {col: int(missing[col]) for col in df.columns if missing[col] > 0}
    metrics["missing_pct"] = {col: float(missing_pct[col]) for col in df.columns if missing_pct[col] > 0}

    distributions = {}
    alerts = []
    for attr in protected_attrs:
        if attr not in df_binned.columns:
            continue
        col_data = df_binned[attr].dropna()
        if pd.api.types.is_numeric_dtype(col_data) and "age" not in attr.lower():
            continue
        vc = col_data.value_counts(normalize=True).round(4)
        dist = {str(k): float(v) for k, v in vc.items()}
        distributions[attr] = dist
        if vc.max() > 0.85:
            alerts.append({
                "type": "imbalance",
                "column": attr,
                "dominant_group": str(vc.idxmax()),
                "pct": float(round(vc.max() * 100, 1))
            })

    metrics["distributions"] = distributions

    target_col = detect_target_column(df)
    outcome_bias = {}
    spd_values = {}
    di_values = {}

    if target_col:
        df_outcome = df_binned.copy()
        df_outcome["_target_binary"] = convert_to_binary(df[target_col])
        metrics["target_column"] = target_col

        for attr in protected_attrs:
            if attr not in df_outcome.columns:
                continue
            col_data = df_outcome[attr].dropna()
            if pd.api.types.is_numeric_dtype(col_data) and "age" not in attr.lower():
                continue
            try:
                grp = df_outcome.groupby(attr)["_target_binary"].mean().round(4)
                group_rates = {str(k): float(v) for k, v in grp.items()}
                if len(group_rates) >= 2:
                    rates = list(group_rates.values())
                    max_rate = max(rates)
                    min_rate = min(rates)
                    spd = round(max_rate - min_rate, 4)
                    di = round(min_rate / max_rate, 4) if max_rate > 0 else 0
                    outcome_bias[attr] = group_rates
                    spd_values[attr] = spd
                    di_values[attr] = di
                    if spd > 0.10 or di < 0.80:
                        alerts.append({
                            "type": "outcome_bias",
                            "column": attr,
                            "target": target_col,
                            "statistical_parity_difference": spd,
                            "disparate_impact": di,
                            "detail": f"Outcome bias detected in '{target_col}' for '{attr}': SPD={spd}, DI={di}",
                            "severity": "HIGH" if (spd > 0.20 or di < 0.60) else "MEDIUM"
                        })
            except Exception:
                pass

    metrics["outcome_bias"] = outcome_bias
    metrics["statistical_parity_difference"] = spd_values
    metrics["disparate_impact"] = di_values
    metrics["alerts"] = alerts
    metrics["alert_count"] = len(alerts)

    # FIX #2: normalized bias score
    metrics["bias_score"] = compute_bias_score(metrics)
    metrics["risk_level"] = score_to_risk(metrics["bias_score"])
    metrics["audit_confidence"] = compute_audit_confidence(metrics)
    return metrics


# ─── Post-Training Metrics ────────────────────────────────────────────────────
def compute_posttraining_metrics(df, prediction_col, label_col=None):
    metrics = {}
    metrics["total_predictions"] = int(len(df))
    metrics["prediction_col"] = prediction_col

    if prediction_col not in df.columns:
        metrics["error"] = f"Prediction column '{prediction_col}' not found"
        return metrics

    preds = df[prediction_col]
    unique_vals = preds.dropna().unique()

    binary_map = {}
    if set(str(v).lower() for v in unique_vals).issubset({"yes", "no", "true", "false", "1", "0", "approved", "rejected", "accept", "deny", "positive", "negative"}):
        positive_vals = {"yes", "true", "1", "approved", "accept", "positive"}
        binary_map = {v: 1 if str(v).lower() in positive_vals else 0 for v in unique_vals}
        df = df.copy()
        df["_pred_binary"] = df[prediction_col].map(lambda x: binary_map.get(x, 1 if str(x).lower() in positive_vals else 0))
    else:
        df = df.copy()
        try:
            df["_pred_binary"] = pd.to_numeric(df[prediction_col], errors="coerce").fillna(0).astype(int)
        except Exception:
            df["_pred_binary"] = 1

    pos_rate = float(df["_pred_binary"].mean())
    metrics["positive_rate"] = round(pos_rate, 4)
    outcome_counts = df["_pred_binary"].value_counts().to_dict()
    metrics["outcome_counts"] = {str(k): int(v) for k, v in outcome_counts.items()}

    df_binned = bin_age_column(df)
    protected_attrs = detect_protected_attributes(df_binned)
    metrics["protected_attributes"] = protected_attrs

    group_rates = {}
    spd_values = {}
    di_values = {}

    for attr in protected_attrs:
        if attr not in df_binned.columns:
            continue
        col_data = df_binned[attr].dropna()
        if pd.api.types.is_numeric_dtype(col_data) and "age" not in attr.lower():
            continue
        try:
            grp = df_binned.groupby(attr)["_pred_binary"].mean().round(4)
            gr = {str(k): float(v) for k, v in grp.items()}
            group_rates[attr] = gr
            if len(gr) >= 2:
                rates = list(gr.values())
                spd = max(rates) - min(rates)
                spd_values[attr] = round(spd, 4)
                di = min(rates) / max(rates) if max(rates) > 0 else None
                di_values[attr] = round(di, 4) if di is not None else None
        except Exception:
            pass

    metrics["group_rates"] = group_rates
    metrics["statistical_parity_difference"] = spd_values
    metrics["disparate_impact"] = di_values

    rule_80 = {}
    for attr, di in di_values.items():
        if di is not None:
            rule_80[attr] = "PASS" if di >= 0.8 else "FAIL"
    metrics["rule_80"] = rule_80

    if label_col and label_col in df.columns:
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            labels = df[label_col]
            label_map = {v: 1 if str(v).lower() in {"yes", "true", "1", "approved", "accept", "positive"} else 0 for v in labels.dropna().unique()}
            y_true = labels.map(lambda x: label_map.get(x, int(x) if str(x).isdigit() else 0))
            y_pred = df["_pred_binary"]
            valid = ~(y_true.isna() | y_pred.isna())
            y_true = y_true[valid].astype(int)
            y_pred = y_pred[valid].astype(int)
            metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
            metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            metrics["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        except Exception:
            pass

    # FIX #2: normalized bias score
    metrics["bias_score"] = compute_bias_score(metrics)
    metrics["risk_level"] = score_to_risk(metrics["bias_score"])
    metrics["audit_confidence"] = compute_audit_confidence(metrics)
    return metrics


def compute_statistical_parity(df, attr, target):
    try:
        groups = df.groupby(attr)[target].mean()
        return float(groups.max() - groups.min())
    except Exception:
        return None


def generate_graph_data(metrics, mode):
    graphs = []
    if mode == "pre":
        dists = metrics.get("distributions", {})
        for attr, dist in dists.items():
            graphs.append({
                "type": "bar",
                "title": f"{attr} Distribution",
                "labels": list(dist.keys()),
                "data": [round(v * 100, 1) for v in dist.values()],
                "attr": attr
            })
        missing = metrics.get("missing_pct", {})
        if missing:
            graphs.append({
                "type": "bar",
                "title": "Missing Values (%)",
                "labels": list(missing.keys()),
                "data": list(missing.values()),
                "attr": "missing"
            })
    elif mode == "post":
        group_rates = metrics.get("group_rates", {})
        for attr, gr in group_rates.items():
            graphs.append({
                "type": "bar",
                "title": f"Approval Rate by {attr}",
                "labels": list(gr.keys()),
                "data": [round(v * 100, 1) for v in gr.values()],
                "attr": attr
            })
        outcome = metrics.get("outcome_counts", {})
        if outcome:
            graphs.append({
                "type": "pie",
                "title": "Outcome Distribution",
                "labels": ["Positive", "Negative"],
                "data": [outcome.get("1", 0), outcome.get("0", 0)],
                "attr": "outcome"
            })
        spd = metrics.get("statistical_parity_difference", {})
        if spd:
            graphs.append({
                "type": "bar",
                "title": "Statistical Parity Difference by Attribute",
                "labels": list(spd.keys()),
                "data": [v if v is not None else 0 for v in spd.values()],
                "attr": "spd"
            })
    return graphs


# ─── Sandbox helpers ──────────────────────────────────────────────────────────
def explain_sandbox_with_groq(approval_by_group, bias_score, protected_attr):
    if not approval_by_group:
        return ""
    rates_str = ", ".join([f"{g}: {int(v*100)}%" for g, v in approval_by_group.items()])
    max_rate = max(approval_by_group.values())
    min_rate = min(approval_by_group.values())
    max_group = max(approval_by_group, key=approval_by_group.get)
    min_group = min(approval_by_group, key=approval_by_group.get)
    gap = max_rate - min_rate
    is_balanced = gap < 0.10
    risk_level = score_to_risk(bias_score)

    provider = get_ai_provider()
    if not provider:
        return ""

    prompt = f"""You are an AI fairness auditor reviewing a counterfactual bias test.

PYTHON-COMPUTED FACTS (use exactly as given):
- Protected attribute: {protected_attr}
- Group approval rates: {rates_str}
- Highest group: {max_group} at {int(max_rate*100)}%
- Lowest group: {min_group} at {int(min_rate*100)}%
- Absolute gap: {round(gap*100, 1)}%
- Bias score: {bias_score}/100
- Risk: {risk_level}
- Balance: {"BALANCED — gap under 10%, no significant counterfactual bias" if is_balanced else f"IMBALANCED — {round(gap*100,1)}% gap detected"}

RULES:
1. Use only these numbers. Never invent values.
2. If balanced -> MUST say no significant bias.
3. 1-2 sentences per section only.

Return this format exactly:

Key Finding:
[State whether changing {protected_attr} changed predictions.]

Evidence:
[Exact approval rates for each group.]

Affected Group:
[{min_group} if gap >= 10%, else "No affected group — gap is within acceptable range."]

Risk:
[State risk with reason.]

Recommendation:
[One specific next step.]"""

    try:
        if provider == "groq" and GROQ_LIB_AVAILABLE:
            client = groq_lib.Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        elif provider == "anthropic" and ANTHROPIC_LIB_AVAILABLE:
            client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
    except Exception:
        traceback.print_exc()
    return ""


def generate_sandbox_with_groq(protected_attr, domain="hiring"):
    provider = get_ai_provider()
    if provider:
        prompt = f"""Generate a biased synthetic Python predict(row) function for testing a fairness auditor in {domain}.
Protected attribute: {protected_attr}
Rules: Only def predict(row):. No imports. Return 1=approved, 0=rejected. Different outcomes for GroupA/B/C/D. Max 12 lines.
Respond ONLY in JSON with no markdown, no backticks, no explanation:
{{"code": "def predict(row):\\n    ...", "sample_json": {{"{protected_attr}": "GroupA", "income": 55000, "score": 70}}, "protected_attr": "{protected_attr}"}}"""
        try:
            if provider == "groq" and GROQ_LIB_AVAILABLE:
                client = groq_lib.Groq(api_key=GROQ_API_KEY)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.3
                )
                raw = response.choices[0].message.content.strip()
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
            elif provider == "anthropic" and ANTHROPIC_LIB_AVAILABLE:
                client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = message.content[0].text.strip()
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
        except Exception:
            traceback.print_exc()
    return {
        "code": f"""def predict(row):
    group = row.get('{protected_attr}', 'GroupA')
    income = float(row.get('income', 50000))
    rates = {{'GroupA': 0.8, 'GroupB': 0.6, 'GroupC': 0.4, 'GroupD': 0.2}}
    threshold = rates.get(group, 0.5)
    return 1 if income > 50000 * (1 / threshold) else 0""",
        "sample_json": {protected_attr: "GroupA", "income": 55000, "score": 70},
        "protected_attr": protected_attr
    }


def explain_optimizer_with_groq(optimizer_result):
    before = optimizer_result.get("before_score", 0)
    after = optimizer_result.get("after_score", 0)
    improvement = optimizer_result.get("improvement", 0)
    attr = optimizer_result.get("protected_attr", "attribute")
    before_dist = optimizer_result.get("before_dist", {})
    after_dist = optimizer_result.get("after_dist", {})
    before_str = ", ".join([f"{g}: {round(v*100,1)}%" for g, v in before_dist.items()])
    after_str = ", ".join([f"{g}: {round(v*100,1)}%" for g, v in after_dist.items()])

    provider = get_ai_provider()
    if not provider:
        return ""

    prompt = f"""You are an AI fairness auditor reviewing a bias optimization simulation.

PYTHON FACTS (use exactly):
- Attribute: {attr}
- Before distribution: {before_str}
- After distribution: {after_str}
- Bias score before: {before}/100 -> after: {after}/100
- Improvement: {improvement} points

RULES: Use only these numbers. No civic/political language. 1-2 sentences each.

Return exactly:

Bias Found:
[Explain original imbalance with exact values.]

Optimization Effect:
[Explain score change.]

Affected Group:
[Groups remaining underrepresented after optimization.]

Reliability Check:
[Is simulation realistic or unstable?]

Recommendation:
[One practical next step.]"""

    try:
        if provider == "groq" and GROQ_LIB_AVAILABLE:
            client = groq_lib.Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        elif provider == "anthropic" and ANTHROPIC_LIB_AVAILABLE:
            client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
    except Exception:
        traceback.print_exc()
    return ""


# ─── AI Report Generator ──────────────────────────────────────────────────────
def generate_ai_report(metrics, mode):
    provider = get_ai_provider()
    if not provider:
        return {"error": "No AI provider configured. Set GROQ_API_KEY or ANTHROPIC_API_KEY in .env"}

    mode_label = {"pre": "Pre-Training Dataset Audit", "post": "Post-Training Model Audit", "appeal": "Appeal Engine Report"}.get(mode, "Fairness Audit")

    prompt = f"""You are a senior AI fairness auditor for a hiring/selection AI system.
Audit type: {mode_label}
COMPUTED METRICS:
{json.dumps(metrics, indent=2)}

STRICT RULES:
- Use ONLY the provided metrics.
- Do NOT invent numbers, causes, laws, or external facts.
- Do NOT use civic outreach, public participation, or political language.
- Do NOT blame protected groups.
- Focus on dataset quality, approval rates, SPD, DI, 80% rule, and bias risk.
- If SPD > 0.10 -> meaningful disparity. If SPD > 0.20 -> severe disparity.
- If DI < 0.80 -> fails 80% fairness rule.
- If bias_score >= 60 -> final verdict must be CRITICAL.
- If key metrics are missing -> INCONCLUSIVE.

Return EXACTLY these 9 sections:

1. EXECUTIVE SUMMARY
2. DATASET/MODEL COMPOSITION
3. MOST AFFECTED GROUPS
4. BIAS RISK ANALYSIS
5. FAIRNESS METRIC FINDINGS
6. LEGAL & ETHICAL IMPLICATIONS
7. FUTURE RISK ASSESSMENT
8. RECOMMENDATIONS
- rebalance underrepresented groups
- review labels for historical bias
- tune decision thresholds per fairness policy
- retrain using fairness constraints
- monitor group-level approval rates
- require human review for high-impact decisions

9. FINAL VERDICT
Return only one: PASS, INCONCLUSIVE, FAIL, CRITICAL.
Then 2-3 sentences explaining why.

Each section: 3-5 sentences max. Plain text only. No markdown headings."""

    try:
        if provider == "groq":
            client = groq_lib.Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            return {"report": response.choices[0].message.content, "provider": "groq"}
        elif provider == "anthropic":
            client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"report": message.content[0].text, "provider": "anthropic"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── What-If Simulations ──────────────────────────────────────────────────────
def run_what_if_pretraining(df, protected_attr, desired_balance):
    if protected_attr not in df.columns:
        return {"error": f"Column '{protected_attr}' not found"}
    df = bin_age_column(df)
    col = df[protected_attr].dropna()
    before_dist = col.value_counts(normalize=True).round(4).to_dict()
    before_dist = {str(k): float(v) for k, v in before_dist.items()}
    temp_metrics_before = compute_pretraining_metrics(df)
    before_score = temp_metrics_before["bias_score"]

    groups = list(before_dist.keys())
    n = len(groups)
    if n == 0:
        return {"error": "No groups found"}
    target = float(desired_balance) / 100
    leftover = 1.0 - target
    after_dist = {}
    majority = max(before_dist, key=before_dist.get)
    for g in groups:
        if g == majority:
            after_dist[g] = round(target, 4)
        else:
            after_dist[g] = round(leftover / (n - 1), 4) if n > 1 else 0.0

    df_sim = df.copy()
    df_sim[protected_attr] = majority
    temp_metrics_after = compute_pretraining_metrics(df_sim)
    after_score = temp_metrics_after["bias_score"]
    improvement = max(0, round(before_score - after_score, 1))
    return {
        "protected_attr": protected_attr,
        "before_dist": before_dist,
        "after_dist": after_dist,
        "before_score": before_score,
        "after_score": after_score,
        "improvement": improvement,
        "groups": groups
    }


def run_what_if_posttraining(df, threshold, fairness_weight, protected_attr):
    if "_pred_binary" not in df.columns:
        df = df.copy()
        for col in df.columns:
            if col.lower() in ("prediction", "decision", "outcome", "label", "result", "approved"):
                positive_vals = {"yes", "true", "1", "approved", "accept", "positive"}
                df["_pred_binary"] = df[col].map(lambda x: 1 if str(x).lower() in positive_vals else 0)
                break
        if "_pred_binary" not in df.columns:
            return {"error": "No prediction column found for optimization"}

    before_rate = float(df["_pred_binary"].mean())
    df2 = df.copy()
    if fairness_weight > 0.5:
        df2["_pred_binary"] = 1
    elif fairness_weight < 0.3:
        df2["_pred_binary"] = df["_pred_binary"]
    after_rate = float(df2["_pred_binary"].mean())

    before_metrics = compute_posttraining_metrics(df, df.columns[0])
    after_metrics = compute_posttraining_metrics(df2, df.columns[0])

    return {
        "before_score": before_metrics.get("bias_score", 0),
        "after_score": after_metrics.get("bias_score", 0),
        "before_positive_rate": round(before_rate, 4),
        "after_positive_rate": round(after_rate, 4),
        "improvement": max(0, round(before_metrics.get("bias_score", 0) - after_metrics.get("bias_score", 0), 1)),
        "threshold": threshold,
        "fairness_weight": fairness_weight
    }


# ─── Stress Testing ───────────────────────────────────────────────────────────
def run_blackbox_stress_test_api(api_url, token, sample_json, protected_attr):
    if not api_url:
        return {"error": "API URL required"}
    try:
        base_payload = json.loads(sample_json) if isinstance(sample_json, str) else sample_json
    except Exception:
        return {"error": "Invalid sample JSON"}

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    synthetic_groups = ["GroupA", "GroupB", "GroupC", "GroupD"]
    results = []
    for group in synthetic_groups:
        payload = dict(base_payload)
        payload[protected_attr] = group
        try:
            resp = http_requests.post(api_url, json=payload, headers=headers, timeout=10)
            decision = resp.json()
        except Exception as e:
            decision = {"error": str(e)}
        results.append({"group": group, "payload": payload, "response": decision})

    approval_by_group = {}
    for r in results:
        resp = r["response"]
        approved = 0
        if isinstance(resp, dict):
            for key in ["approved", "decision", "result", "prediction", "label"]:
                if key in resp:
                    approved = 1 if str(resp[key]).lower() in {"true", "yes", "1", "approved", "accept", "positive"} else 0
                    break
        approval_by_group[r["group"]] = approved

    rates = list(approval_by_group.values())
    if len(rates) >= 2 and max(rates) > 0:
        spd = max(rates) - min(rates)
        di = min(rates) / max(rates)
        temp_metrics = {"statistical_parity_difference": {"g": spd}, "disparate_impact": {"g": di}}
        bias_score = compute_bias_score(temp_metrics)
    else:
        bias_score = 0

    return {
        "results": results,
        "approval_by_group": approval_by_group,
        "bias_score": round(bias_score, 1),
        "risk_level": score_to_risk(bias_score),
        "counterfactual_rate": round(bias_score, 1)
    }


def run_blackbox_stress_test_sandbox(code, sample_json, protected_attr):
    if not code or not code.strip():
        return {"error": "No code provided"}
    try:
        base_payload = json.loads(sample_json) if isinstance(sample_json, str) else sample_json
    except Exception:
        return {"error": "Invalid sample JSON"}

    synthetic_profiles = [
        {**base_payload, protected_attr: "GroupA"},
        {**base_payload, protected_attr: "GroupB"},
        {**base_payload, protected_attr: "GroupC"},
        {**base_payload, protected_attr: "GroupD"},
    ]

    runner_code = f"""
import json, sys
ALLOWED_BUILTINS = {{'print': print, 'range': range, 'len': len, 'int': int,
                    'float': float, 'str': str, 'bool': bool, 'list': list,
                    'dict': dict, 'abs': abs, 'round': round, 'max': max,
                    'min': min, 'sum': sum, 'enumerate': enumerate, 'zip': zip}}
user_code = {json.dumps(code)}
profiles = {json.dumps(synthetic_profiles)}
try:
    ns = {{'__builtins__': ALLOWED_BUILTINS}}
    exec(user_code, ns)
    predict = ns.get('predict')
    if predict is None:
        print(json.dumps({{'error': 'No predict() function found'}}))
        sys.exit(1)
    results = []
    for row in profiles:
        try:
            result = predict(row)
            results.append({{'group': row.get('{protected_attr}', 'unknown'), 'result': result}})
        except Exception as e:
            results.append({{'group': row.get('{protected_attr}', 'unknown'), 'error': str(e)}})
    print(json.dumps({{'results': results}}))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(runner_code)
        tmp_path = f.name

    try:
        import sys as _sys
        proc = subprocess.run([_sys.executable, tmp_path], capture_output=True, text=True, timeout=10)
        result_data = json.loads(proc.stdout.strip())
    except subprocess.TimeoutExpired:
        result_data = {"error": "Sandbox timeout (10s)"}
    except Exception as e:
        result_data = {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if "error" in result_data:
        return result_data

    results = result_data.get("results", [])
    approval_by_group = {}
    for r in results:
        group = r.get("group", "unknown")
        result = r.get("result", 0)
        approved = 1 if str(result).lower() in {"true", "yes", "1", "approved", "accept", "positive"} or result == 1 else 0
        approval_by_group[group] = approved

    rates = list(approval_by_group.values())
    if len(rates) >= 2 and max(rates) > 0:
        spd = max(rates) - min(rates)
        di = min(rates) / max(rates)
        temp_metrics = {"statistical_parity_difference": {"g": spd}, "disparate_impact": {"g": di}}
        bias_score = compute_bias_score(temp_metrics)
    else:
        bias_score = 0

    groq_explanation = explain_sandbox_with_groq(approval_by_group, bias_score, protected_attr)

    return {
        "results": results,
        "approval_by_group": approval_by_group,
        "bias_score": round(bias_score, 1),
        "risk_level": score_to_risk(bias_score),
        "counterfactual_rate": round(bias_score, 1),
        "profiles_tested": len(synthetic_profiles),
        "groq_explanation": groq_explanation
    }


# ─── Appeal Engine ────────────────────────────────────────────────────────────
def run_appeal_engine(document_text, policy_text, domain):
    provider = get_ai_provider()
    if not provider:
        return {"error": "No AI provider configured"}

    prompt = f"""You are an AI appeal analyst for a hiring/selection decision.
Domain: {domain}
USER DOCUMENT:
{document_text[:3000]}
POLICY / CRITERIA:
{policy_text[:3000]}
STRICT RULES:
- Use only the user document and policy text.
- Do NOT invent achievements, skills, reasons, or missing requirements.
- Be neutral and non-discriminatory.
- Focus on policy match, evidence gaps, and appeal quality.

Return EXACTLY this format:
FIT SCORE: [0-100]
MATCHED REQUIREMENTS:
- [requirement + evidence]
MISSING REQUIREMENTS:
- [requirement not found]
LIKELY REJECTION REASONS:
- [policy-based reasons only]
POSSIBLE FAIRNESS CONCERNS:
- [only if relevant; otherwise "none identified"]
IMPROVEMENT PLAN:
- [specific improvements]
APPEAL RECOMMENDATION:
[Strong Appeal / Moderate Appeal / Weak Appeal / No Appeal Recommended]
APPEAL SUMMARY:
[2-3 sentences.]
No markdown. No invented facts."""

    try:
        if provider == "groq":
            client = groq_lib.Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            report_text = response.choices[0].message.content
        elif provider == "anthropic":
            client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            report_text = message.content[0].text

        fit_score = 50
        for line in report_text.split("\n"):
            if line.strip().startswith("FIT SCORE:"):
                try:
                    fit_score = int(line.split(":")[1].strip().split()[0])
                except Exception:
                    pass

        return {"report": report_text, "fit_score": fit_score, "provider": provider}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not MONGO_AVAILABLE:
            session["user_id"] = "demo_user"
            session["user_email"] = email
            session["user_name"] = email.split("@")[0]
            return redirect(url_for("dashboard"))
        user = db["users"].find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user_id"] = str(user["_id"])
            session["user_email"] = email
            session["user_name"] = user.get("name", email.split("@")[0])
            return redirect(url_for("dashboard"))
        flash("Invalid email or password", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not MONGO_AVAILABLE:
            session["user_id"] = "demo_user"
            session["user_email"] = email
            session["user_name"] = name
            return redirect(url_for("dashboard"))
        if db["users"].find_one({"email": email}):
            flash("Email already registered", "error")
            return render_template("register.html")
        user_id = db["users"].insert_one({
            "name": name, "email": email,
            "password": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc)
        }).inserted_id
        session["user_id"] = str(user_id)
        session["user_email"] = email
        session["user_name"] = name
        return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    provider = get_ai_provider()
    return render_template("dashboard.html",
                           user_name=session.get("user_name", "User"),
                           ai_provider=provider,
                           mongo_available=MONGO_AVAILABLE)


# ─── Pre-Training API ─────────────────────────────────────────────────────────
@app.route("/api/pretrain", methods=["POST"])
@login_required
def api_pretrain():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    mode, data = load_uploaded_file(file)
    if mode == "error":
        return jsonify({"error": data}), 400

    # ── TEXT MODE: TXT / PDF / XML / HTML that couldn't be parsed as a DataFrame
    # Run the full pipeline using text-extraction metrics so graphs/alerts/report all work.
    if mode == "text":
        filename_safe = secure_filename(file.filename) if file.filename else "document"
        metrics = compute_text_metrics(data, filename_safe)
        graphs = generate_text_graph_data(metrics)

        # Graph explanations
        groq_explanations = []
        for g in graphs:
            explanation = explain_graph_with_groq({"title": g["title"], "labels": g["labels"], "data": g["data"]})
            groq_explanations.append({"chart": g["title"], "explanation": explanation})

        # Alert explanations
        target_col = metrics.get("target_column")
        alert_explanations = []
        for alert in metrics.get("alerts", []):
            explanation = generate_alert_explanation_groq(alert, target_col)
            alert_explanations.append({
                "column": alert.get("column"),
                "type": alert.get("type"),
                "severity": alert.get("severity", "HIGH"),
                "explanation": explanation,
                "spd": alert.get("statistical_parity_difference"),
                "di": alert.get("disparate_impact"),
                "dominant_group": alert.get("dominant_group"),
                "pct": alert.get("pct"),
                "target": alert.get("target"),
            })

        report_result = generate_ai_report(metrics, "pre")
        result = {
            "mode": "text_analysis",
            "metrics": metrics,
            "graphs": graphs,
            "groq_explanations": groq_explanations,
            "alert_explanations": alert_explanations,
            "report": report_result.get("report", ""),
            "report_error": report_result.get("error"),
            "provider": report_result.get("provider", ""),
            "filename": filename_safe,
            "message": f"Analysed as unstructured text ({metrics['file_type'].upper()}). "
                       f"Demographic mentions extracted: {metrics['text_stats']['demographic_mentions']}. "
                       f"Full bias metrics computed from co-occurrence analysis."
        }

        if MONGO_AVAILABLE:
            doc = normalize_for_mongo({
                "user_id": session["user_id"],
                "mode": "pre-training",
                "filename": filename_safe,
                "metrics": metrics,
                "graphs": graphs,
                "groq_explanations": groq_explanations,
                "alert_explanations": alert_explanations,
                "report": report_result.get("report", ""),
                "provider": report_result.get("provider", ""),
                "bias_score": metrics.get("bias_score", 0),
                "risk_level": metrics.get("risk_level", "UNKNOWN"),
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            inserted = db["reports"].insert_one(doc)
            result["report_id"] = str(inserted.inserted_id)

        return jsonify(normalize_for_mongo(result))

    df = data
    metrics = compute_pretraining_metrics(df)
    graphs = generate_graph_data(metrics, "pre")

    # Graph explanations via Groq
    groq_explanations = []
    for g in graphs:
        explanation = explain_graph_with_groq({"title": g["title"], "labels": g["labels"], "data": g["data"]})
        groq_explanations.append({"chart": g["title"], "explanation": explanation})

    # FIX #1 + #3: Rich Groq-powered alert explanations (stored in DB)
    target_col = metrics.get("target_column")
    alert_explanations = []
    for alert in metrics.get("alerts", []):
        explanation = generate_alert_explanation_groq(alert, target_col)
        alert_explanations.append({
            "column": alert.get("column"),
            "type": alert.get("type"),
            "severity": alert.get("severity", "HIGH"),
            "explanation": explanation,
            "spd": alert.get("statistical_parity_difference"),
            "di": alert.get("disparate_impact"),
            "dominant_group": alert.get("dominant_group"),
            "pct": alert.get("pct"),
            "target": alert.get("target"),
        })

    report_result = generate_ai_report(metrics, "pre")
    result = {
        "metrics": metrics,
        "graphs": graphs,
        "groq_explanations": groq_explanations,
        "alert_explanations": alert_explanations,
        "report": report_result.get("report", ""),
        "report_error": report_result.get("error"),
        "provider": report_result.get("provider", ""),
        "filename": secure_filename(file.filename) if file.filename else "unknown"
    }

    # FIX #3: Store ALL data in MongoDB including groq explanations and alert explanations
    if MONGO_AVAILABLE:
        doc = normalize_for_mongo({
            "user_id": session["user_id"],
            "mode": "pre-training",
            "filename": result["filename"],
            "metrics": metrics,
            "graphs": graphs,
            "groq_explanations": groq_explanations,       # FIX #3: now stored
            "alert_explanations": alert_explanations,          # FIX #3: now stored
            "report": report_result.get("report", ""),
            "provider": report_result.get("provider", ""),
            "bias_score": metrics.get("bias_score", 0),
            "risk_level": metrics.get("risk_level", "UNKNOWN"),
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        inserted = db["reports"].insert_one(doc)
        result["report_id"] = str(inserted.inserted_id)

    return jsonify(normalize_for_mongo(result))


# ─── Post-Training API ────────────────────────────────────────────────────────
@app.route("/api/posttrain", methods=["POST"])
@login_required
def api_posttrain():
    prediction_col = request.form.get("prediction_col", "").strip()
    label_col = request.form.get("label_col", "").strip() or None
    df = None

    if "file" in request.files and request.files["file"].filename:
        mode, data = load_uploaded_file(request.files["file"])
        if mode == "error":
            return jsonify({"error": data}), 400
        if mode == "text":
            return jsonify({"error": "Please upload CSV/JSON/XLSX.", "switch_to_text": True}), 400
        df = data
    elif "file_data" in request.files and "file_pred" in request.files:
        mode1, data1 = load_uploaded_file(request.files["file_data"])
        mode2, data2 = load_uploaded_file(request.files["file_pred"])
        if mode1 != "dataframe" or mode2 != "dataframe":
            return jsonify({"error": "Both files must be CSV/JSON/XLSX"}), 400
        if len(data1) != len(data2):
            return jsonify({"error": f"Row count mismatch: data={len(data1)}, predictions={len(data2)}"}), 400
        df = pd.concat([data1, data2], axis=1)
    else:
        return jsonify({"error": "No file uploaded"}), 400

    if not prediction_col:
        candidates = ["prediction", "decision", "outcome", "label", "result", "approved", "predicted"]
        for col in df.columns:
            if col.lower() in candidates:
                prediction_col = col
                break
        if not prediction_col:
            return jsonify({"error": "prediction_col not specified. Available columns: " + ", ".join(df.columns)}), 400

    metrics = compute_posttraining_metrics(df, prediction_col, label_col)
    if "error" in metrics:
        return jsonify({"error": metrics["error"]}), 400

    graphs = generate_graph_data(metrics, "post")

    # Graph explanations via Groq
    groq_explanations = []
    for g in graphs:
        explanation = explain_graph_with_groq({"title": g["title"], "labels": g["labels"], "data": g["data"]})
        groq_explanations.append({"chart": g["title"], "explanation": explanation})

    # FIX #1 + #3: Rich Groq-powered alert explanations for post-training fairness issues
    post_alerts = []
    for attr, di in metrics.get("disparate_impact", {}).items():
        spd = metrics.get("statistical_parity_difference", {}).get(attr, 0)
        if di is not None and (di < 0.80 or (spd and spd > 0.10)):
            post_alerts.append({
                "type": "outcome_bias",
                "column": attr,
                "target": prediction_col,
                "statistical_parity_difference": spd,
                "disparate_impact": di,
                "severity": "HIGH" if (di < 0.60 or spd > 0.20) else "MEDIUM"
            })

    alert_explanations = []
    for alert in post_alerts:
        explanation = generate_alert_explanation_groq(alert, prediction_col)
        alert_explanations.append({
            "column": alert.get("column"),
            "type": alert.get("type"),
            "severity": alert.get("severity", "HIGH"),
            "explanation": explanation,
            "spd": alert.get("statistical_parity_difference"),
            "di": alert.get("disparate_impact"),
            "target": alert.get("target"),
        })

    report_result = generate_ai_report(metrics, "post")
    filename = request.files.get("file", request.files.get("file_data")).filename

    result = {
        "metrics": metrics,
        "graphs": graphs,
        "groq_explanations": groq_explanations,
        "alert_explanations": alert_explanations,
        "report": report_result.get("report", ""),
        "report_error": report_result.get("error"),
        "provider": report_result.get("provider", ""),
        "filename": secure_filename(filename) if filename else "unknown",
        "df_columns": list(df.columns)
    }

    # FIX #3: Store ALL data in MongoDB including groq explanations and alert explanations
    if MONGO_AVAILABLE:
        doc = normalize_for_mongo({
            "user_id": session["user_id"],
            "mode": "post-training",
            "filename": result["filename"],
            "metrics": metrics,
            "graphs": graphs,
            "groq_explanations": groq_explanations,       # FIX #3: now stored
            "alert_explanations": alert_explanations,          # FIX #3: now stored
            "report": report_result.get("report", ""),
            "provider": report_result.get("provider", ""),
            "bias_score": metrics.get("bias_score", 0),
            "risk_level": metrics.get("risk_level", "UNKNOWN"),
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        inserted = db["reports"].insert_one(doc)
        result["report_id"] = str(inserted.inserted_id)

    return jsonify(normalize_for_mongo(result))


# ─── What-If Routes ───────────────────────────────────────────────────────────
@app.route("/api/whatif/pre", methods=["POST"])
@login_required
def api_whatif_pre():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data sent"}), 400
    df_data = data.get("df_data")
    protected_attr = data.get("protected_attr", "")
    desired_balance = data.get("desired_balance", 50)
    if not df_data:
        return jsonify({"error": "No dataset data provided"}), 400
    try:
        df = pd.DataFrame(df_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    result = run_what_if_pretraining(df, protected_attr, desired_balance)
    if "error" not in result:
        result["groq_explanation"] = explain_optimizer_with_groq(result)
    return jsonify(normalize_for_mongo(result))

@app.route("/api/whatif/post", methods=["POST"])
@login_required
def api_whatif_post():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data sent"}), 400
    df_data = data.get("df_data")
    threshold = data.get("threshold", 0.5)
    fairness_weight = data.get("fairness_weight", 0.5)
    protected_attr = data.get("protected_attr", "")
    if not df_data:
        return jsonify({"error": "No dataset data provided"}), 400
    try:
        df = pd.DataFrame(df_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    result = run_what_if_posttraining(df, threshold, fairness_weight, protected_attr)
    return jsonify(normalize_for_mongo(result))

@app.route("/api/whatif/explain", methods=["POST"])
@login_required
def api_whatif_explain():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data sent"}), 400
    explanation = explain_optimizer_with_groq(data)
    return jsonify({"explanation": explanation})


# ─── Stress Testing Routes ────────────────────────────────────────────────────
@app.route("/api/stress/api", methods=["POST"])
@login_required
def api_stress_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data sent"}), 400
    result = run_blackbox_stress_test_api(
        data.get("api_url", ""), data.get("token", ""),
        data.get("sample_json", "{}"), data.get("protected_attr", "gender")
    )
    return jsonify(normalize_for_mongo(result))

@app.route("/api/stress/sandbox", methods=["POST"])
@login_required
def api_stress_sandbox():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data sent"}), 400
    result = run_blackbox_stress_test_sandbox(
        data.get("code", ""), data.get("sample_json", "{}"), data.get("protected_attr", "gender")
    )
    return jsonify(normalize_for_mongo(result))

@app.route("/api/stress/generate", methods=["POST"])
@login_required
def api_stress_generate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data sent"}), 400
    result = generate_sandbox_with_groq(data.get("protected_attr", "gender"), data.get("domain", "hiring"))
    return jsonify(result)


# ─── Appeal Engine Route ──────────────────────────────────────────────────────
@app.route("/api/appeal", methods=["POST"])
@login_required
def api_appeal():
    document_text = ""
    policy_text = ""
    domain = request.form.get("domain", "hiring")

    if "file_doc" in request.files and request.files["file_doc"].filename:
        mode, data = load_uploaded_file(request.files["file_doc"])
        if mode in ("text", "dataframe"):
            document_text = data if mode == "text" else data.to_string()
        else:
            return jsonify({"error": data}), 400
    else:
        document_text = request.form.get("doc_text", "").strip()

    policy_url = request.form.get("policy_url", "").strip()
    if policy_url:
        try:
            resp = http_requests.get(policy_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            policy_text = resp.text[:5000]
        except Exception as e:
            policy_text = f"Could not fetch URL: {e}"
    elif "file_policy" in request.files and request.files["file_policy"].filename:
        mode, data = load_uploaded_file(request.files["file_policy"])
        policy_text = data if mode in ("text", "dataframe") else ""
    else:
        policy_text = request.form.get("policy_text", "").strip()

    if not document_text:
        return jsonify({"error": "Document text is required"}), 400
    if not policy_text:
        return jsonify({"error": "Policy text or URL is required"}), 400

    result = run_appeal_engine(document_text, policy_text, domain)
    if "error" in result:
        return jsonify(result), 500

    if MONGO_AVAILABLE:
        doc = normalize_for_mongo({
            "user_id": session["user_id"], "mode": "appeal", "domain": domain,
            "report": result.get("report", ""), "fit_score": result.get("fit_score", 0),
            "provider": result.get("provider", ""), "bias_score": 100 - result.get("fit_score", 50),
            "risk_level": "MEDIUM", "filename": "appeal",
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        inserted = db["reports"].insert_one(doc)
        result["report_id"] = str(inserted.inserted_id)

    return jsonify(result)


# ─── Reports Routes ───────────────────────────────────────────────────────────
@app.route("/api/reports")
@login_required
def api_reports():
    if not MONGO_AVAILABLE:
        return jsonify({"reports": [], "message": "MongoDB unavailable"})
    try:
        reports = list(db["reports"].find(
            {"user_id": session["user_id"]},
            {"report": 0, "metrics": 0, "graphs": 0}
        ).sort("created_at", -1).limit(50))
        for r in reports:
            r["_id"] = str(r["_id"])
        return jsonify({"reports": reports})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reports/<report_id>")
@login_required
def api_report_detail(report_id):
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB unavailable"}), 503
    try:
        report = db["reports"].find_one({"_id": ObjectId(report_id), "user_id": session["user_id"]})
        if not report:
            return jsonify({"error": "Report not found"}), 404
        report["_id"] = str(report["_id"])
        return jsonify(normalize_for_mongo(report))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reports/<report_id>/rename", methods=["PATCH"])
@login_required
def api_report_rename(report_id):
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB unavailable"}), 503
    try:
        data = request.get_json()
        new_name = (data or {}).get("name", "").strip()
        if not new_name:
            return jsonify({"error": "Name cannot be empty"}), 400
        result = db["reports"].update_one(
            {"_id": ObjectId(report_id), "user_id": session["user_id"]},
            {"$set": {"filename": new_name}}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Report not found"}), 404
        return jsonify({"ok": True, "filename": new_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reports/<report_id>", methods=["DELETE"])
@login_required
def api_report_delete(report_id):
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB unavailable"}), 503
    try:
        result = db["reports"].delete_one(
            {"_id": ObjectId(report_id), "user_id": session["user_id"]}
        )
        if result.deleted_count == 0:
            return jsonify({"error": "Report not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5000)