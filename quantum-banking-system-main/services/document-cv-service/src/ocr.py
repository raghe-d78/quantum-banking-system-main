"""
OCR module & field extraction (spec §5) — v2.

What changed compared with v1 (all of it aimed at "no date" and "always amount_mismatch"):

  image   OCR runs on an upscaled, illumination-normalised image (not the denoised/CLAHE one), in several
          passes (grey + adaptive-binarised, psm 6 + psm 11). It stops as soon as date and amount are found,
          and retries on a second source image (the un-warped original) if the first one yields nothing.
  lines   words are re-grouped into lines by geometry, not by Tesseract's block/line ids (psm 11 makes every
          fragment its own block, which broke both the regexes and layout_consistency).
  date    tolerant to OCR confusions (O/0, l/I/1), space / "|" separators, 2-digit years, "12 mai 2024",
          compact "12052024". Dates are masked out before amounts / document numbers are searched.
  amount  every candidate keeps ALL its plausible readings ("1.250" = 1250 or 1.25), RIB / long digit runs are
          masked, and amount_match() accepts the declared amount if ANY candidate reading matches. A mismatch is
          only reported when the best candidate is a confident, currency/hint-anchored reading; otherwise the
          result is "unknown" (0.5) instead of a false 0.0.
  conf    confidence of a field = mean confidence of the words overlapping its character span
          (v1 used `w.text in needle`, which matched every 1-char word).
"""
from __future__ import annotations
import difflib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import cv2
import numpy as np

from .config import TESSERACT_LANGS

log = logging.getLogger("cv.ocr")

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
OCR_TARGET_SIDE = int(os.environ.get("OCR_TARGET_SIDE", "2400"))   # long side fed to Tesseract
OCR_BUDGET_S = float(os.environ.get("OCR_BUDGET_S", "15"))         # soft wall-clock budget for extra passes
PASSES = [("gray", 6), ("gray", 11), ("bin", 6), ("bin", 11)]

BANKS = ["BIAT", "BNA", "STB", "ATTIJARI", "AMEN BANK", "UIB", "BH BANK", "ZITOUNA", "ATB", "UBCI", "BTK", "QNB", "BANQUE", "BANK"]
CURRENCY_RE = re.compile(r"(?<![A-Za-z])(TND|DT|DINARS?|EUROS?|EUR|€|USD|\$)(?![A-Za-z])", re.I)
CUR_AFTER_RE = re.compile(r"\s*(?:TND|DT|D\.T|DINARS?|EUROS?|EUR|€|USD|\$)(?![A-Za-z])", re.I)
CUR_BEFORE_RE = re.compile(r"(?:TND|DT|DINARS?|EUROS?|EUR|€|USD|\$)\s*[#*=|\[\]]*\s*$", re.I)
AMOUNT_HINT_BEFORE_RE = re.compile(r"(?:montant|amount|somme|pour|total)\s*[:\-]?\s*[#*=]*\s*$", re.I)
DECOR_BEFORE_RE = re.compile(r"[#*=]{1,3}\s*$")
DECOR_AFTER_RE = re.compile(r"\s*[#*=]{1,3}")
AMOUNT_RE = re.compile(r"(?<![\d.,/])(?:\d{1,3}(?:[ .,]\d{3})+(?:[.,]\d{1,3})?|\d+(?:[.,]\d{1,3})?)(?![\d/])")
LONG_DIGITS_RE = re.compile(r"\d(?:[ ]?\d){13,}")          # RIB / IBAN fragments (14+ digits)
BIG_INT_RE = re.compile(r"(?<![\d.,])\d{8,}(?![\d])")       # cheque / account numbers: never an amount
DOCNO_RE = re.compile(r"(?<![\d.,])(\d{6,12})(?![\d.,])")
DOCNO_HINT_RE = re.compile(r"N\s*[°ºo]?\s*[:.]?\s*(\d{6,12})(?!\d)", re.I)
PAYEE_HINTS = [r"pay(?:ez)?\s+(?:to\s+the\s+order\s+of|à\s+l['’]ordre\s+de|a\s+l['’]ordre\s+de|to(?!\s+the))\s*[:\-]?\s*(.*)",
               r"(?:the\s+)?order\s+of\s*[:\-]?\s*(.*)", r"ordre\s+de\s*[:\-]?\s*(.*)", r"b[ée]n[ée]ficiaire\s*[:\-]?\s*(.*)"]

# ── dates ─────────────────────────────────────────────────────────
MONTHS = {
    "janvier": 1, "janv": 1, "jan": 1, "january": 1,
    "février": 2, "fevrier": 2, "fév": 2, "fev": 2, "feb": 2, "february": 2,
    "mars": 3, "mar": 3, "march": 3, "avril": 4, "avr": 4, "apr": 4, "april": 4, "mai": 5, "may": 5,
    "juin": 6, "jun": 6, "june": 6, "juillet": 7, "juil": 7, "jul": 7, "july": 7,
    "août": 8, "aout": 8, "aug": 8, "august": 8, "septembre": 9, "sept": 9, "sep": 9, "september": 9,
    "octobre": 10, "oct": 10, "october": 10, "novembre": 11, "nov": 11, "november": 11,
    "décembre": 12, "decembre": 12, "déc": 12, "dec": 12, "december": 12,
}
_D = r"[0-9OoIl|]"                                   # digit-ish glyphs OCR confuses with digits
_SEP = r"(?:\s*[/.\-\\|]\s*|\s+)"
_MON = "|".join(sorted((re.escape(k) for k in MONTHS), key=len, reverse=True))
DMY_RE = re.compile(rf"(?<!\w)(?P<d>{_D}{{1,2}}){_SEP}(?P<m>{_D}{{1,2}}){_SEP}(?P<y>{_D}{{4}}|{_D}{{2}})(?!\w)")
YMD_RE = re.compile(rf"(?<!\w)(?P<y>{_D}{{4}}){_SEP}(?P<m>{_D}{{1,2}}){_SEP}(?P<d>{_D}{{1,2}})(?!\w)")
_LET = r"[A-Za-zÀ-ÿ]"
DMONY_RE = re.compile(rf"(?<!\w)(?P<d>\d{{1,2}})(?:er|st|nd|rd|th)?\s*(?P<mon>{_LET}{{3,9}})\.?,?\s*(?P<y>\d{{4}}|\d{{2}})(?!\d)")
MDY_RE = re.compile(rf"(?<![A-Za-zÀ-ÿ])(?P<mon>{_LET}{{3,9}})\.?\s*(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(?P<y>\d{{4}})(?!\d)")
_MONTH_KEYS = [k for k in MONTHS if len(k) >= 4]


def _month(tok: str) -> int | None:
    """Month number from a (French/English, possibly OCR-damaged) month word: 'july', 'Luly', 'octobre', 'Dec'."""
    t = tok.lower().strip(".")
    if t in MONTHS:
        return MONTHS[t]
    if len(t) >= 4:
        m = difflib.get_close_matches(t, _MONTH_KEYS, n=1, cutoff=0.74)
        if m:
            return MONTHS[m[0]]
    return None
COMPACT_RE = re.compile(r"(?<!\d)(?P<d>\d{2})(?P<m>\d{2})(?P<y>(?:19|20)\d{2})(?!\d)")
DATE_HINT_RE = re.compile(r"(?:\ble|date|tunis|sfax|sousse|\bà|\ba)\W*$", re.I)
_GLYPHS = str.maketrans("OoIl|", "00111")


@dataclass
class Word:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    line: tuple


@dataclass
class AmountCand:
    text: str
    values: list          # plausible numeric readings, most likely first
    conf: float
    priority: int         # 4 re-read of the amount box, 3 hint/currency, 2 #-anchored, 1 has separators, 0 bare integer


@dataclass
class OCRResult:
    available: bool
    words: list = field(default_factory=list)
    text: str = ""
    fields: dict = field(default_factory=dict)
    confidences: dict = field(default_factory=dict)
    mean_confidence: float = 0.0
    amount_candidates: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    img: object = None          # prepared image the words' coordinates refer to (for the amount-box re-read)


# ── image preparation + Tesseract ─────────────────────────────────
def _prep(img: np.ndarray, variant: str) -> np.ndarray:
    """Upscale to ~OCR_TARGET_SIDE, flatten uneven lighting / background tint, optionally binarise."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = min(3.0, OCR_TARGET_SIDE / max(h, w))
    if abs(s - 1.0) > 0.05:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    k = max(15, (max(g.shape) // 60) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = cv2.divide(g, np.maximum(bg, 1), scale=255)             # text → dark, paper → ~255
    if variant == "bin":
        out = cv2.adaptiveThreshold(cv2.GaussianBlur(out, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 41, 12)
    return cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def run_tesseract(img: np.ndarray, psm: int) -> list[Word]:
    import pytesseract
    data = pytesseract.image_to_data(img, lang=TESSERACT_LANGS, config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
                                     output_type=pytesseract.Output.DICT)
    return _assign_lines(_words_from_data(data))


def _words_from_data(data: dict) -> list[Word]:
    words = []
    for i, txt in enumerate(data["text"]):
        t = (txt or "").strip()
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < 0:
            continue
        words.append(Word(t, conf / 100.0, int(data["left"][i]), int(data["top"][i]), int(data["width"][i]),
                          int(data["height"][i]), (0, 0, 0)))
    return words


def _assign_lines(words: list[Word]) -> list[Word]:
    """Group words into text lines by vertical centre (independent of Tesseract's block/par/line ids)."""
    if not words:
        return words
    hs = float(np.median([w.h for w in words])) or 1.0
    lines: list[list[Word]] = []
    cur: list[Word] = []
    cy = 0.0
    for w in sorted(words, key=lambda w: w.y + w.h / 2):
        c = w.y + w.h / 2
        if cur and abs(c - cy) > 0.6 * hs:
            lines.append(cur); cur = []
        cur.append(w)
        cy = float(np.mean([x.y + x.h / 2 for x in cur]))
    if cur:
        lines.append(cur)
    for i, ln in enumerate(lines):
        for w in ln:
            w.line = (0, 0, i)
    return words


def _lines(words: list[Word]) -> list[list[Word]]:
    out: dict = {}
    for w in words:
        out.setdefault(w.line, []).append(w)
    return [sorted(v, key=lambda w: w.x) for _, v in sorted(out.items())]


def _join(ln: list[Word]) -> tuple[str, list]:
    txt, spans = "", []
    for w in ln:
        if txt:
            txt += " "
        spans.append((len(txt), len(txt) + len(w.text), w))
        txt += w.text
    return txt, spans


def _conf_span(spans: list, a: int, b: int) -> float:
    cs = [w.conf for s, e, w in spans if s < b and e > a]
    return float(np.mean(cs)) if cs else 0.0


def _mask(txt: str, a: int, b: int) -> str:
    return txt[:a] + " " * (b - a) + txt[b:]


# ── dates ─────────────────────────────────────────────────────────
def _num(s: str) -> int | None:
    if not any(ch.isdigit() for ch in s):
        return None
    try:
        return int(s.translate(_GLYPHS))
    except ValueError:
        return None


def date_matches(txt: str, today: date | None = None) -> list[tuple[date, int, int, int]]:
    """All plausible dates in a line: (date, start, end, score). Score favours 'le …/date …' context."""
    today = today or date.today()
    out = []
    for rx, kind in ((DMY_RE, "dmy"), (YMD_RE, "ymd"), (DMONY_RE, "dmony"), (MDY_RE, "mdy"), (COMPACT_RE, "compact")):
        for m in rx.finditer(txt):
            if kind in ("dmony", "mdy"):
                d, mo, y = int(m["d"]), _month(m["mon"]), _num(m["y"])
            else:
                d, mo, y = _num(m["d"]), _num(m["m"]), _num(m["y"])
            if d is None or mo is None or y is None:
                continue
            if y < 100:
                y += 2000
            if not (today.year - 15 <= y <= today.year + 1):
                continue
            try:
                dt = date(y, mo, d)
            except ValueError:
                continue
            score = (2 if DATE_HINT_RE.search(txt[max(0, m.start() - 14):m.start()]) else 0) + (0 if kind == "compact" else 1)
            out.append((dt, m.start(), m.end(), score))
    return out


# ── amounts ───────────────────────────────────────────────────────
def amount_variants(s: str) -> list[float]:
    """Every plausible numeric reading of an OCR'd amount string, most likely first.
    '1.250,500'→[1250.5]  '1250,500'→[1250.5]  '1.250'→[1250, 1.25]  '250,000'→[250.0, 250000]  '1 250'→[1250]"""
    s = re.sub(r"\s+", " ", s.strip(" #*=-:_~"))
    if not re.fullmatch(r"\d[\d .,]*", s):
        return []
    if " " in s:
        if not re.fullmatch(r"\d{1,3}(?: \d{3})+(?:[.,]\d{1,3})?", s):
            return []
        s = s.replace(" ", "")
    vals: list[float] = []
    try:
        if "," in s and "." in s:
            dec = "," if s.rfind(",") > s.rfind(".") else "."
            thou = "." if dec == "," else ","
            ip, _, fp = s.rpartition(dec)
            ip = ip.replace(thou, "")
            if ip.isdigit() and fp.isdigit() and len(fp) <= 3:
                vals = [float(f"{ip}.{fp}")]
        elif "," in s or "." in s:
            sep = "," if "," in s else "."
            parts = s.split(sep)
            if len(parts) > 2:
                if all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
                    vals = [float("".join(parts))]
            elif parts[0] and parts[1]:
                head, tail = parts
                dec_v = float(f"{head}.{tail}")
                if len(tail) == 3 and 1 <= len(head) <= 3:          # ambiguous: grouping or 3-digit millimes
                    thou_v = float(head + tail)
                    vals = [thou_v, dec_v] if sep == "." else [dec_v, thou_v]
                else:
                    vals = [dec_v]
        else:
            vals = [float(s)]
    except ValueError:
        return []
    return [v for v in vals if 0 < v < 1e9]


def parse_amount(s: str) -> float | None:          # backward-compatible helper
    v = amount_variants(s)
    return v[0] if v else None


def find_amounts(txt: str, spans: list) -> list[AmountCand]:
    out = []
    for m in AMOUNT_RE.finditer(txt):
        vals = amount_variants(m.group(0))
        if not vals:
            continue
        before, after = txt[max(0, m.start() - 16):m.start()], txt[m.end():m.end() + 8]
        has_sep = bool(re.search(r"[.,]", m.group(0)))
        digits = len(re.sub(r"\D", "", m.group(0)))
        if (AMOUNT_HINT_BEFORE_RE.search(before) or CUR_AFTER_RE.match(after) or CUR_BEFORE_RE.search(before)) and (has_sep or digits >= 3):
            pri = 3                                    # "montant 500" / "DT 500,000" / "$ 500.00" / "500 TND"
        elif has_sep and (DECOR_BEFORE_RE.search(before) or DECOR_AFTER_RE.match(after)):
            pri = 2
        elif has_sep:
            pri = 1
        else:
            pri = 0                                    # bare integers (MICR line, cheque no., years) never outrank the rest
        out.append(AmountCand(m.group(0).strip(), vals, _conf_span(spans, m.start(), m.end()), pri))
    return out


def merge_amounts(cands: list[AmountCand]) -> list[AmountCand]:
    best: dict = {}
    for c in cands:
        k = tuple(c.values)
        if k not in best or (c.priority, c.conf) > (best[k].priority, best[k].conf):
            best[k] = c
    return sorted(best.values(), key=lambda c: (c.priority, bool(re.search(r"[.,]", c.text)), c.conf), reverse=True)


# ── field extraction ──────────────────────────────────────────────
def extract_fields(words: list[Word], today: date | None = None) -> tuple[dict, dict, list[AmountCand]]:
    lines = [_join(ln) for ln in _lines(words)]
    full = "\n".join(t for t, _ in lines)
    fields: dict = {}
    conf: dict = {}

    # dates first, so their digits can be masked out of everything else
    best_date = None
    masked: list[tuple[str, list]] = []
    for txt, spans in lines:
        mt = txt
        for dt, a, b, score in date_matches(txt, today):
            c = _conf_span(spans, a, b)
            if best_date is None or (score, c) > (best_date[1], best_date[2]):
                best_date = (dt, score, c)
            mt = _mask(mt, a, b)
        masked.append((mt, spans))
    if best_date:
        fields["date"], conf["date"] = best_date[0].isoformat(), best_date[2]

    # currency
    m = CURRENCY_RE.search(full)
    if m:
        cur = m.group(1).upper()
        fields["currency"] = "TND" if cur in ("DT", "TND") or cur.startswith("DINAR") else ("EUR" if cur.startswith("EUR") or cur == "€" else "USD")
        for txt, spans in lines:
            i = CURRENCY_RE.search(txt)
            if i:
                conf["currency"] = _conf_span(spans, i.start(), i.end()); break

    # amounts (dates, RIB-like runs and 8+ digit numbers masked)
    cands: list[AmountCand] = []
    for mt, spans in masked:
        t = mt
        for rx in (LONG_DIGITS_RE, BIG_INT_RE):
            for mm in list(rx.finditer(t)):
                t = _mask(t, mm.start(), mm.end())
        cands += find_amounts(t, spans)
    cands = merge_amounts(cands)
    if cands:
        fields["amount"], conf["amount"] = round(cands[0].values[0], 3), cands[0].conf

    # document number (prefer "N° 1234567")
    for mt, spans in masked:
        nm = DOCNO_HINT_RE.search(mt) or DOCNO_RE.search(re.sub(r"\d(?:[ ]?\d){13,}", " ", mt))
        if nm:
            fields["document_number"] = nm.group(1)
            conf["document_number"] = _conf_span(spans, nm.start(1), nm.end(1))
            if DOCNO_HINT_RE.search(mt):
                break

    # payee
    def _clean_name(raw: str) -> str:
        name = re.sub(r"[^A-Za-zÀ-ÿ' &.-]", " ", raw).strip()
        return re.split(r"\s{2,}|\s(?:montant|amount|somme|TND|DT)\b", name, 1, flags=re.I)[0].strip()

    for li, (txt, spans) in enumerate(lines):
        for hint in PAYEE_HINTS:
            pm = re.search(hint, txt, re.I)
            if not pm:
                continue
            name, c = _clean_name(pm.group(1)), _conf_span(spans, pm.start(1), pm.end(1))
            if len(name) < 2 and li + 1 < len(lines):            # label alone on its line: the name is on the next one
                nt, ns = lines[li + 1]
                nn = _clean_name(nt)
                if len(nn) >= 2 and sum(ch.isalpha() for ch in nt) >= 0.6 * len(nt.replace(" ", "")):
                    name, c = nn, _conf_span(ns, 0, len(nt))
            if 2 <= len(name) <= 60:
                fields["payee"], conf["payee"] = name, c
                break
        if "payee" in fields:
            break

    # bank
    up = full.upper()
    for b in BANKS:
        if b in up:
            fields["bank"] = b
            for txt, spans in lines:
                i = txt.upper().find(b.split()[0])
                if i >= 0:
                    conf["bank"] = _conf_span(spans, i, i + len(b.split()[0])); break
            break
    return fields, conf, cands


def _build(words: list[Word], meta: dict, today: date | None) -> OCRResult:
    fields, confs, cands = extract_fields(words, today)
    scored = [w.conf for w in words if len(w.text) >= 2 or w.text.isdigit()]       # ignore 1-char noise (borders, stamps)
    mean_conf = float(np.mean(scored)) if scored else 0.0
    r = OCRResult(available=True, words=words, text="\n".join(" ".join(w.text for w in ln) for ln in _lines(words)),
                  fields=fields, confidences={k: round(v, 4) for k, v in confs.items()},
                  mean_confidence=round(mean_conf, 4), amount_candidates=cands, meta=dict(meta))
    r.meta["score"] = round(2 * ("date" in fields) + 2 * bool(cands) + ("currency" in fields) + ("document_number" in fields)
                            + ("payee" in fields) + mean_conf, 3)
    return r


def ocr(images, today: date | None = None, budget_s: float = OCR_BUDGET_S) -> OCRResult:
    """images: one BGR/gray array or a list of them (best source first). Runs several passes, keeps the best
    and fills any field it is missing from the other passes."""
    if not TESSERACT_AVAILABLE:
        return OCRResult(available=False)
    if isinstance(images, np.ndarray):
        images = [images]
    t0 = time.monotonic()
    results: list[OCRResult] = []
    errors = runs = 0
    done = False
    for si, img in enumerate(images):
        for variant, psm in PASSES:
            if runs and time.monotonic() - t0 > budget_s:
                done = True; break
            runs += 1
            try:
                prepped = _prep(img, variant)
                words = run_tesseract(prepped, psm)
            except Exception as e:  # noqa: BLE001
                errors += 1; log.warning("tesseract pass failed (%s/psm%s): %s", variant, psm, e); continue
            if not words:
                continue
            r = _build(words, {"source": si, "variant": variant, "psm": psm}, today)
            r.img = prepped
            results.append(r)
            top = max(results, key=lambda x: x.meta["score"])
            if "date" in top.fields and any(c.priority >= 2 for c in top.amount_candidates) and top.mean_confidence >= 0.6:
                done = True; break
        if done:
            break
    if not results:
        return OCRResult(available=errors < runs, meta={"passes_run": runs})
    results.sort(key=lambda x: x.meta["score"], reverse=True)
    best = results[0]
    for r in results[1:]:
        for k in ("date", "currency", "document_number", "payee", "bank"):
            if k not in best.fields and k in r.fields:
                best.fields[k] = r.fields[k]; best.confidences[k] = r.confidences.get(k, 0.0)
    merged = merge_amounts([c for r in results for c in r.amount_candidates])
    try:
        boxed = amount_box_candidates(best.img, best.words) if best.img is not None else []
    except Exception as e:  # noqa: BLE001
        boxed = []; log.warning("amount-box re-read failed: %s", e)
    best.meta["amount_box_reads"] = len(boxed)
    merged = merge_amounts(merged + boxed)
    best.amount_candidates = merged
    if merged:
        best.fields["amount"], best.confidences["amount"] = round(merged[0].values[0], 3), round(merged[0].conf, 4)
    best.meta.update({"passes_run": runs, "passes_used": len(results)})
    best.img = None
    return best


# ── amount-box re-read ────────────────────────────────────────────
CUR_TOKEN_RE = re.compile(r"^\W{0,2}(?:TND|DT|D\.T|PT|OT|0T|EUR|USD)\W{0,2}$", re.I)


def _is_cur_token(t: str) -> bool:
    return bool(CUR_TOKEN_RE.match(t)) or (len(t) <= 4 and ("$" in t or "€" in t))


def _strip_lines(b: np.ndarray) -> np.ndarray:
    """Remove long vertical/horizontal rules (the amount box's border) from a black-on-white binary crop."""
    inv = 255 - b
    h, w = inv.shape
    v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.55)))))
    z = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * 0.6)), 1)))
    rules = cv2.dilate(cv2.bitwise_or(v, z), np.ones((3, 3), np.uint8))
    return 255 - cv2.subtract(inv, rules)


def _read_digits(crop: np.ndarray) -> list[AmountCand]:
    """Digits-only OCR of a small single-line crop (grey and rule-stripped binary), zoomed to ~110 px high."""
    import pytesseract
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if g.shape[0] < 8 or g.shape[1] < 20:
        return []
    g = cv2.resize(g, None, fx=min(6.0, 110.0 / g.shape[0]), fy=min(6.0, 110.0 / g.shape[0]), interpolation=cv2.INTER_CUBIC)
    b = _strip_lines(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    reads: list[list[AmountCand]] = []
    for im in (g, b):
        out = []
        im = cv2.copyMakeBorder(im, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        data = pytesseract.image_to_data(im, lang="eng", config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,",
                                         output_type=pytesseract.Output.DICT)
        txt, spans = _join(sorted(_words_from_data(data), key=lambda w: w.x))
        for m in AMOUNT_RE.finditer(txt):
            vals = amount_variants(m.group(0))
            if vals and len(re.sub(r"\D", "", m.group(0))) >= 2:
                out.append(AmountCand(m.group(0).strip(), vals, _conf_span(spans, m.start(), m.end()), 4))
        reads.append(out)
    # Tesseract reports confidence 0 when a whitelist is set, so confidence = agreement between the two reads.
    agreed = set.intersection(*({tuple(c.values) for c in r} for r in reads)) if all(reads) else set()
    flat = [c for r in reads for c in r]
    for c in flat:
        c.conf = 0.9 if tuple(c.values) in agreed else max(c.conf, 0.55)
    return flat


def amount_box_candidates(img: np.ndarray, words: list[Word], max_tokens: int = 2) -> list[AmountCand]:
    """Find currency tokens (DT / TND …) and re-read the area on each side of them: the printed amount box."""
    H, W = img.shape[:2]
    out: list[AmountCand] = []
    for w in [w for w in words if _is_cur_token(w.text)][:max_tokens]:
        hh = max(w.h, 20)
        y0, y1 = max(0, w.y - int(0.6 * hh)), min(H, w.y + w.h + int(0.6 * hh))
        for x0, x1 in ((w.x + w.w, min(W, w.x + w.w + 12 * hh)), (max(0, w.x - 12 * hh), w.x)):
            if x1 - x0 >= 40:
                out += _read_digits(img[y0:y1, x0:x1])
    return out


# ── cross-checks ──────────────────────────────────────────────────
def date_validity(doc_date_iso: str | None, today: date | None = None, stale_days: int = 180) -> tuple[float, str]:
    """1.0 valid, 0.5 unknown, 0.0 post-dated or stale. Returns (score, reason)."""
    if not doc_date_iso:
        return 0.5, "no_date"
    today = today or date.today()
    try:
        d = datetime.fromisoformat(doc_date_iso).date()
    except ValueError:
        return 0.5, "unparseable"
    if d > today:
        return 0.0, "post_dated"
    if (today - d).days > stale_days:
        return 0.0, "stale"
    return 1.0, "valid"


def amount_match(candidates, expected: float | None, tol: float = 0.01, min_conf: float = 0.6) -> tuple[float, str]:
    """1.0 if ANY candidate reading matches the declared amount, 0.5 unknown/unverified, 0.0 confident mismatch.
    `candidates` may also be a plain float (legacy)."""
    if expected is None or expected <= 0:
        return 0.5, "unknown"
    if isinstance(candidates, (int, float)):
        candidates = [AmountCand(str(candidates), [float(candidates)], 1.0, 2)]
    if not candidates:
        return 0.5, "no_amount_read"
    for c in candidates:
        for v in c.values:
            if abs(v - expected) / expected <= tol:
                return 1.0, "match"
    top = max(candidates, key=lambda c: (c.priority, c.conf))
    if top.priority >= 2 and top.conf >= min_conf:
        return 0.0, f"mismatch(read {top.values[0]:g}, declared {expected:g})"
    return 0.5, "unverified"