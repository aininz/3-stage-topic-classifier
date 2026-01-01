from __future__ import annotations

import pandas as pd

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Type


# -------------------------------------------------
# Base text cleaner
# -------------------------------------------------

class TextCleaner:
    CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")
    MULTISPACE_RE = re.compile(r"\s+")
    URL_RE = re.compile(r"https?://\S+|www\.\S+")

    # “stuck together” patterns
    CAMEL_1 = re.compile(r"([a-z])([A-Z])")
    CAMEL_2 = re.compile(r"([A-Z])([A-Z][a-z])")
    PUNCT_WORD = re.compile(r"([.!?,;:])([A-Za-z])")

    @staticmethod
    def safe_str(x) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x).strip()
        return "" if s.lower() == "nan" else s

    @classmethod
    def fix_missing_spaces(cls, s: str) -> str:
        """
        Inserts spaces in cases like:
        - camelCase -> camel Case
        - ABCDef -> ABC Def
        - "hello.World" -> "hello. World"
        """
        s = cls.safe_str(s)
        if not s:
            return ""

        s = cls.CAMEL_1.sub(r"\1 \2", s)
        s = cls.CAMEL_2.sub(r"\1 \2", s)
        s = cls.PUNCT_WORD.sub(r"\1 \2", s)
        return s

    @classmethod
    def normalize(cls, s: str, *, fix_spaces: bool = True) -> str:
        s = cls.safe_str(s)
        if not s:
            return ""

        s = unicodedata.normalize("NFKC", s)
        s = cls.CONTROL_CHARS_RE.sub(" ", s)
        s = cls.URL_RE.sub(" ", s)
        s = s.replace("\n", " ").replace("\r", " ")

        if fix_spaces:
            s = cls.fix_missing_spaces(s)

        s = cls.MULTISPACE_RE.sub(" ", s).strip()
        return s

    @classmethod
    def clean(cls, s: str) -> str:
        return cls.normalize(s)


# -------------------------------------------------
# Phone numbers cleaner
# -------------------------------------------------

class PhoneNumberCleaner(TextCleaner):
    """
    Normalize phone-number-like substrings in text.

    Examples:
      "+62 812 345 678"   -> "+62812345678"
      "(+1) 415-555-2671" -> "+14155552671"
      "0049 30 1234 567"  -> "+49301234567"
      "0812-345-6789"     -> "08123456789"

    Notes:
    - Country-agnostic: keeps '+' if present; converts leading '00' -> '+'.
    - Removes separators: spaces, hyphens, dots, parentheses.
    - Preserves '*' if already present (masked inputs).
    """

    PHONE_CANDIDATE_RE = re.compile(
        r"""
        (?<!\w)
        (?:\+|00)?\s*
        (?:\(?\d{1,4}\)?[\s\-.]*)?
        (?:\d[\s\-.()]*){7,}
        (?:\*+)?
        (?!\w)
        """,
        re.VERBOSE
    )

    KEEP_DIGITS_PLUS_STAR_RE = re.compile(r"[^\d\+\*]+")

    @classmethod
    def _normalize_one(cls, m: re.Match) -> str:
        raw = m.group(0)
        s = cls.KEEP_DIGITS_PLUS_STAR_RE.sub("", raw)

        if s.count("+") > 1:
            s = "+" + s.replace("+", "")

        if s.startswith("00"):
            s = "+" + s[2:]

        if "+" in s[1:]:
            s = "+" + s.replace("+", "")

        digits_only = re.sub(r"[^\d]", "", s)
        if len(digits_only) < 8:
            return raw

        return s

    @classmethod
    def clean(cls, s: str) -> str:
        s = cls.normalize(s, fix_spaces=False)
        if not s:
            return s
        return cls.PHONE_CANDIDATE_RE.sub(cls._normalize_one, s)


# -------------------------------------------------
# News cleaners
# -------------------------------------------------

class NewsCleaner(TextCleaner, ABC):
    """
    Generic news cleaner: provides a stripping pipeline engine.
    Subclasses should define get_steps() and patterns.
    """

    @classmethod
    def _apply(cls, s: str, pat: re.Pattern, repl: str = " ") -> str:
        s = pat.sub(repl, s).strip()
        return cls.normalize(s)

    @classmethod
    def _strip_repeat(cls, s: str, pat: re.Pattern, repl: str = " ", max_iter: int = 3) -> str:
        for _ in range(max_iter):
            s2 = pat.sub(repl, s).strip()
            s2 = cls.normalize(s2)
            if s2 == s:
                break
            s = s2
        return s

    @classmethod
    @abstractmethod
    def get_steps(cls):
        """
        Return an iterable of steps:
          (kind, pattern, replacement, max_iter)
        kind in {"apply","repeat"}.
        """
        raise NotImplementedError

    @classmethod
    def clean(cls, s: str) -> str:
        s = cls.normalize(s)
        if not s:
            return s

        for kind, pat, repl, max_iter in cls.get_steps():
            if kind == "apply":
                s = cls._apply(s, pat, repl)
            elif kind == "repeat":
                s = cls._strip_repeat(s, pat, repl=repl, max_iter=max_iter or 3)
            else:
                raise ValueError(f"Unknown step kind: {kind}")

            if not s:
                break

        return s


class IndonesianNewsCleaner(NewsCleaner):
    LEADING_JUNK_RE = re.compile(r"^\s*[-–—:|]+\s*")

    BOILERPLATE_RE = re.compile(
        r"\bBaca berita dengan sedikit iklan,\s*klik di sini\b",
        re.IGNORECASE
    )

    SECTION_CAPS_RE = re.compile(
        r"""^\s*[A-Z][A-Z\s]{2,40}\s*(?:[-–—:|•·]\s*)+""",
        re.VERBOSE
    )

    NEWS_PREFIX_RE = re.compile(
        r"""^\s*
        (?:[A-Z][A-Z\s\.\-]{1,50}?,\s*)?
        (?:
            KOMPAS(?:\.com)?|
            TEMPO(?:\.CO)?|
            DETIK(?:\.com)?|
            CNNINDONESIA(?:\.com)?|
            TRIBUNNEWS(?:\.com)?|
            LIPUTAN6(?:\.com)?|
            ANTARA(?:NEWS)?|
            OKEZONE(?:\.com)?|
            KONTAN(?:\.co\.id)?|
            INILAH(?:\.com)?|
            SINDO(?:NEWS)?(?:\.com)?|
            SUARA(?:\.com)?|
            MERDEKA(?:\.com)?|
            IDN(?:TIMES)?(?:\.com)?
        )
        \s*(?:,?\s*[A-Za-z][A-Za-z\s\.\-]{1,30})?
        \s*(?:[-–—:|•·]\s*)+
        """,
        re.VERBOSE | re.IGNORECASE
    )

    BOILERPLATE_UI_RE = re.compile(
        r"\b(?:bawah\s+melanjutkan|scroll\s+bawah|lanjutkan\s+membaca|baca\s+selengkapnya|klik\s+di\s+sini)\b",
        re.IGNORECASE
    )

    @classmethod
    def get_steps(cls):
        return (
            ("apply",  cls.BOILERPLATE_RE, " ", None),
            ("apply",  cls.BOILERPLATE_UI_RE, " ", None),
            ("repeat", cls.SECTION_CAPS_RE, " ", 3),
            ("repeat", cls.NEWS_PREFIX_RE, " ", 3),
            ("apply",  cls.LEADING_JUNK_RE, "",  None),
        )
    
# -------------------------------------------------
# Finance cleaner
# -------------------------------------------------

class FinanceCleaner(TextCleaner):
    """
    Finance-oriented normalization:
    - percentages: 6.25%, 6,25 persen -> <PCT>
    - currency amounts: Rp 1,2 triliun / USD 2.5B / $120 -> <AMT_IDR>, <AMT_USD>, <AMT_CUR>
    - tickers: $BBCA, BTC, ETH, BTC/USDT -> <TICKER>, <CRYPTO>, <PAIR>
    - (optional) pure numbers -> <NUM> to reduce noise
    """

    # 12,5% | 12.5% | 12 persen | 12,5 persen
    PCT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|persen)\b", re.IGNORECASE)

    # Currency markers (Indonesia + common global)
    # Rp / IDR / USD / US$ / EUR / GBP etc.
    CURRENCY_WORD_RE = re.compile(
        r"\b(?:rp|idr|usd|us\$|cad|ca\$|eur|gbp|aud|sgd|jpy|cny|hkd)\b",
        re.IGNORECASE,
    )

    # $123, $1.2B, $ 1,200.50
    DOLLAR_AMOUNT_RE = re.compile(
        r"(?<!\w)\$\s*\d[\d.,]*(?:\s*[KMBT])?(?!\w)"
    )

    # Rp 1,2 triliun | Rp. 1.200.000 | IDR 3,5 miliar
    IDR_AMOUNT_RE = re.compile(
        r"(?<!\w)(?:rp\.?|idr)\s*\d[\d.,]*\s*(?:ribu|juta|miliar|triliun)?(?!\w)",
        re.IGNORECASE,
    )

    # USD 2.5B | EUR 10 juta | SGD 1,2
    FX_AMOUNT_RE = re.compile(
        r"(?<!\w)(?:usd|us\$|eur|gbp|aud|sgd|jpy|cny|hkd)\s*\d[\d.,]*\s*(?:[KMBT]|ribu|thousand|juta|million|miliar|billion|triliun|trillion)?(?!\w)",
        re.IGNORECASE,
    )

    # Stock/crypto tickers with $ prefix: $BBCA, $BTC
    DOLLAR_TICKER_RE = re.compile(r"(?<!\w)\$[A-Z]{2,10}\b")

    # Crypto pairs: BTC/USDT, ETH-IDR, BTCUSDT (simple)
    PAIR_RE = re.compile(
        r"\b[A-Z]{2,10}\s*[/\-]\s*[A-Z]{2,10}\b"
    )

    # Common crypto tickers
    CRYPTO_RE = re.compile(
        r"\b(?:BTC|ETH|USDT|BNB|SOL|XRP|ADA|DOGE|DOT|MATIC|AVAX|LINK|LTC|BCH|ATOM|ICP)\b",
        re.IGNORECASE,
    )

    # Basis points / bps
    BPS_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:bps|bp)\b", re.IGNORECASE)

    # Pure numbers (optional)
    NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    @classmethod
    def clean(
        cls,
        s: str,
        *,
        mask_numbers: bool = False,
        keep_currency_words: bool = True,
    ) -> str:
        s = cls.normalize(s)
        if not s:
            return s

        # Order matters, so pairs/tickers before amounts sometimes
        s = cls.PAIR_RE.sub(" <PAIR> ", s)
        s = cls.DOLLAR_TICKER_RE.sub(" <TICKER> ", s)
        s = cls.CRYPTO_RE.sub(" <CRYPTO> ", s)

        s = cls.PCT_RE.sub(" <PCT> ", s)
        s = cls.BPS_RE.sub(" <BPS> ", s)

        s = cls.IDR_AMOUNT_RE.sub(" <AMT_IDR> ", s)
        s = cls.FX_AMOUNT_RE.sub(" <AMT_FX> ", s)
        s = cls.DOLLAR_AMOUNT_RE.sub(" <AMT_USD> ", s)

        if not keep_currency_words:
            s = cls.CURRENCY_WORD_RE.sub(" ", s)

        if mask_numbers:
            s = cls.NUM_RE.sub(" <NUM> ", s)

        s = cls.MULTISPACE_RE.sub(" ", s).strip()
        return s


# -------------------------------------------------
# Chaining cleaners
# -------------------------------------------------

@dataclass(frozen=True)
class CleanerPipeline:
    cleaners: tuple[Type["TextCleaner"], ...]

    def __call__(self, text) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            s = ""
        else:
            s = str(text)

        for C in self.cleaners:
            s = C.clean(s)
            if not s:
                break
        return s
