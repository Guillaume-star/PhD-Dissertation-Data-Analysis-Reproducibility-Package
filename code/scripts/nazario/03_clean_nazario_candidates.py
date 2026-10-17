from __future__ import annotations

import argparse
import hashlib
import html
import quopri
import re
import string
from pathlib import Path

import pandas as pd


MIN_BODY_WORDS = 25
MAX_BODY_WORDS = 500

HTML_TAG_RE = re.compile(r"</?\s*[a-z][^>\n]{0,500}>", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
HTML_COMMENT_DELIMITER_RE = re.compile(r"(?://\s*)?(?:<!--|-->)")
ANGLE_ARROW_DECORATION_RE = re.compile(r">{2,}")
STANDALONE_SEPARATOR_LINE_RE = re.compile(r"(?m)^\s*=*\s*-{2,}\s*$")
STANDALONE_DOT_LINE_RE = re.compile(r"(?m)^\s*\.\s*$")
STANDALONE_QUESTION_LINE_RE = re.compile(r"(?m)^\s*\?\s*$")
STANDALONE_UNDERSCORE_LINE_RE = re.compile(r"(?m)^\s*_{2,}\s*$")
STANDALONE_SMILEY_LINE_RE = re.compile(r"(?m)^\s*:\)\s*$")
STANDALONE_GT_LINE_RE = re.compile(r"(?m)^\s*>\s*$")
STANDALONE_QUOTE_LINE_RE = re.compile(r"(?m)^\s*\"\s*$")
LEADING_TARGET_BLANK_RE = re.compile(r"(?is)^\s*[\"')\s,.;:-]*target\s*=\s*_?blank\s*>\s*")
LEADING_COLON_COMMA_RE = re.compile(r"(?m)^\s*[:,]+\s*(?=[A-Za-z])")
LEADING_BODY_LABEL_RE = re.compile(r"(?is)^\s*body\s+(?=(?:dear|\{|[A-Z]))")
LEADING_PLUS_EQUALS_RE = re.compile(r"(?m)^\s*[+=]\s*(?=[A-Za-z])")
SPACED_DHL_RE = re.compile(r"(?im)^\s*D\s+H\s+L\b")
LEADING_ORPHAN_TAG_CLOSE_RE = re.compile(r"(?m)^\s*\">\s*")
DANGLING_TAG_FRAGMENT_RE = re.compile(r"(?im)<\s*/?\s*(?:a|br|span)\b[^>\n]*$")
MALFORMED_FONT_FRAGMENT_RE = re.compile(
    r"<font\s+(?:face|size|color)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>\"]+)\s*",
    re.IGNORECASE,
)
ORPHAN_CSS_STYLE_FRAGMENT_RE = re.compile(
    r"(?im)^\s*;?\s*font-family\s*:\s*[^\"\n>]+\">\s*$"
)
IMP_FOOTER_RE = re.compile(
    r"(?im)^\s*This message was sent using IMP, the Internet Messaging Program\.\s*$"
)
CSS_DECLARATION_RE = re.compile(r"[^{}\n]{0,220}\{[^{}\n]{0,800}\}\s*")
CSS_PROPERTY_RE = re.compile(
    r"\b(?:border|margin|font-family|font-size|font-weight|color|text-decoration|width)\s*:",
    re.IGNORECASE,
)
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,20}|#[0-9]{2,7}|#x[0-9a-fA-F]{2,6});")
QUOTED_PRINTABLE_RE = re.compile(r"(?<![A-Za-z0-9_])=(?:[0-9A-Fa-f]{2}|3D|20|09|0A|0D)|(?<![=])=\s*$", re.MULTILINE)
ISO2022JP_ESCAPE_RE = re.compile(r"\$B|\(B")
DENSE_ENCODED_PAYLOAD_RE = re.compile(
    r"`A8v84|A8v84|AENF6|F5NF6|AE\{B1|\^10b37|g0DRA1|ÿ0C|~E7~ED|Q8D|N2D_C3"
)
LITERARY_PADDING_RE = re.compile(
    r"Ducky Daddles|rocker beginning to move|fill in your own fucking n|"
    r"resurrection man sitting in a pony-trap|cuny alabaster|"
    r"very good today|do you best from|manytoday|bestbest|"
    r"bruce chaw ditto|dentnon|blutwurst|chromatography|californium|"
    r"are wept, lure sue vet|apt whey soletuft|arty oboe, soil horn|"
    r"m snvpfy kwfn|w odvuu|Lkildlu pokmcf|"
    r"btjjqy Ctowr|wjwwhg fjqfs|qwhvj ltvqb|"
    r"bts\s+wa Nnpiu|hmcku ormks|Juaabwj Uimlpyd|"
    r"cup mush, zalot bank|lobe hank, tap catchon|turn pool, cut dime|"
    r"bounty host christine|rocsea -o-alcove|pealdung christmas|"
    r"Un usu y al act|ww d w\. p c h ase|J o se ph A|"
    r"I SELL THE BELOW TOOLS|UNLIMITED SMTP.*SCAMPAGE|BANK LOGING PAGE|"
    r"FREE MOVIE TICKETS|CLAIM YOUR 4 MOVIE TICKETS|LapRunner partner site",
    re.IGNORECASE,
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_LINK_REFERENCE_RE = re.compile(r"\[\d{1,3}\]")
NUMBER_MARKER_RE = re.compile(r"##number\d+##", re.IGNORECASE)
EMPTY_OR_IMAGE_BRACKET_RE = re.compile(r"\[\s*(?:image|img|picture|graphic)?\s*\]", re.IGNORECASE)
HTML_PLAIN_TEXT_NOTICE_RE = re.compile(
    r"(?im)^\s*-?\s*this\s+mail\s+is\s+in\s+html\.\s+"
    r"some\s+elements\s+may\s+be\s+omm?ited\s+in\s+plain\s+text\.\s*-?\s*$"
)
LEADING_TITLE_RE = re.compile(
    r"(?is)^\s*(?:untitled document|documento sin t(?:í|i|√≠)tulo|"
    r"html editor sample page|new page \d+|صفحة\s+جديدة\s*\d*)\s*(?:\n+|$)"
)
HTML_ATTRIBUTE_RE = re.compile(
    r"\b(?:href|src|style|class|width|height|border|cellpadding|cellspacing|font-family|bgcolor|align)=",
    re.IGNORECASE,
)
MOJIBAKE_PATTERNS = ("�", "Â", "Ã", "â€", "â€™", "â€œ", "â€�", "¬†", "ðŸ", "√")
COMMON_MOJIBAKE_REPLACEMENTS = {
    "–ø¬ª—ó": "",
    "–í¬©": "",
    "√Ø¬ª¬ø": "",
    "ï»¿": "",
    "п»ї": "",
    "В©": "",
    "ЁП": "",
    "Ўн": "",
    "ўз": "",
    "ЁЎПн": "",
    "¬Æ": "",
    "¬ª": "",
    "»": "",
    "¬§¬±": "",
    "➡️": "",
    "➡": "",
    "‚û°Ô∏è": "",
    "‚Äã": "",
    "¬†": " ",
    "Â\xa0": " ",
    "Â ": " ",
    "Â": "",
    "â€‹": "",
    "‚Äô": "'",
    "‚Äôs": "'s",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€¢": "-",
    "‚Ä¢": "-",
    "•": "-",
    "ўG 0": "0",
    "ўG": "",
    "Œ°ay Œ°al": "PayPal",
    "Œ°ayŒ°al": "PayPal",
    "Œ°ay": "Pay",
    "Œ°al": "Pal",
    "Œ°lease": "Please",
    "Œúy": "My",
    "Œëccount": "Account",
    "Œóelp": "Help",
    "Œëpps": "Apps",
    "Œïbay": "eBay",
    "Ρay Ρal": "PayPal",
    "ΡayΡal": "PayPal",
    "Ρay": "Pay",
    "Ρal": "Pal",
    "reνiewed": "reviewed",
    "serνice": "service",
    "sensitiνe": "sensitive",
    "inconνenience": "inconvenience",
    "haνe": "have",
    "belieνe": "believe",
    "Βecause": "Because",
    "√Ä": "À",
    "√°": "á",
    "√§": "ä",
    "√©": "é",
    "√®": "è",
    "√≠": "í",
    "√±": "ñ",
    "√≥": "ó",
    "√∂": "ö",
    "√º": "ü",
    "√ü": "ß",
    "Ã©": "e",
    "Ã\x82": "",
    "Ã²": "o",
    "Ã¨": "e",
    "Ã¡": "a",
    "Ã³": "o",
    "Ã±": "n",
    "Ã¼": "u",
    "®": "",
    "©": "",
}
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
NON_ENGLISH_MARKER_RE = re.compile(
    r"\b("
    r"bonjour|veuillez|cliquez|compte|vérification|présent|impératif|"
    r"cher|client|caisse|epargne|honneur|disposition|syst[eè]me|"
    r"s[eé]curit[eé]|activer|quelques|informations|merci|"
    r"facture|courrier|électronique|equipement|équipement|environnementale|"
    r"orange|"
    r"caro|prezado|usu[aá]rio|atualiza[cç][aã]o|seguran[cç]a|senha|conta|"
    r"informa[cç][oõ]es|pagamento|d[eé]bito|vencimento|fatura|necessario|"
    r"voc[eê]|recebeu|coment[aá]rio|grava[cç][aã]o|ouvir|inscri[cç][aã]o|"
    r"estimado|contrase[nñ]a|verificaci[oó]n|informaci[oó]n|seguridad|"
    r"banco|servicio|haga|clic|aqu[ií]|datos|rellenar|"
    r"loteria|espa[nñ]ol|castellana|videojuego|interactivo|"
    r"regalo|guantes|consola|inalambrico|diversion|"
    r"obligatorio|leerlo|confirme|verifica|cuenta|t[eé]cnico|renov[oó]|"
    r"sehr|geehrt(?:er|e|es)?|kunde|kundin|bankkonto|kontoinhaber|"
    r"betr[uü]gerei(?:en)?|betrgerei(?:en)?|betrgerisch(?:en)?|"
    r"zus[aä]tzliche|zustzliche|autorisierung|aktivierung|"
    r"verwaltung|mitteilung|abteilung|rechte|vorbehalten|"
    r"deutsche|postbank|vrbank|schutzsystem|schutzystem|"
    r"technische|modernisierung|m[uü]ssen|mssen|daf[uü]r|dafr|"
    r"sch[aä]tzen|schtzen|gro[sß]e|groe|"
    r"guten|tag|kontoaktivitaeten|rechnung|abrechnungszeitraum|"
    r"faelliger|saeumnisgebuehren|gruessen|zusaetzliche|"
    r"achtung|gesch[aä]tzten|stillgelegt|gel[oö]scht|"
    r"mail-server|postfach|best[aä]tigen|"
    r"v[aá]žen[yý]|uživateli|e-mailov[eé]ho|účtu|současn[eé]|"
    r"ověřuje|odběratelů|heslo|spr[aá]vce|syst[eé]mu|"
    r"gentile|posteitaliane|banca|conto|codice|sicurezza|"
    r"anagrafici|sottoscrizione|contrattuale|accesso|"
    r"misure|avviso|informarti|trattamento"
    r")\b",
    re.IGNORECASE,
)
STRONG_NON_ENGLISH_PHRASE_RE = re.compile(
    r"("
    r"sehr\s+geehrt|deutsche\s+postbank|rechte\s+sind\s+vorbehalten|"
    r"guten\s+tag[\s\S]{0,250}kontoaktivitaeten|"
    r"(?:ebay\s+international\s+ag[\s\S]{0,250}rechnung|rechnung[\s\S]{0,250}ebay\s+international\s+ag)|"
    r"saeumnisgebuehren|mit\s+freundlichen\s+gruessen|"
    r"cher\(e\)\s+client\(e\)|caisse\s+d['’]epargne|"
    r"service\s+paypal\s+a\s+l['’]honneur|"
    r"votre\s+disposition\s+un\s+nouveau\s+syst[eè]me|"
    r"pour\s+l['’]activer|merci\s+de\s+cliquer|"
    r"merci\s+de\s+votre\s+confiance|votre\s+service\s+clients\s+internet|"
    r"merci\s+de\s+ne\s+pas\s+r[eé]pondre|contribution\s+environnementale|"
    r"gentile\s+cliente|posteitaliane|banca\s+di\s+roma|"
    r"estimado\s+cliente|servicio\s+t[eé]cnico|confirme\s+sus\s+datos|"
    r"verifica\s+tu\s+cuenta|firmen\s+la\s+carta|"
    r"euromillion\s+loteria|paseo\s+de\s+la\s+castellana|"
    r"videojuego\s+virtual\s+interactivo|especial\s+dia\s+del\s+ni[nñ]o|"
    r"voc[eê]\s+recebeu\s+um\s+coment[aá]rio|ouvir\s+coment[aá]rio|"
    r"cancelar\s+inscri[cç][aã]o|"
    r"proteger\s+su\s+salud|delfines\s+inocentes|"
    r"west\s+union\s+group|coordinador\s+de\s+recursos\s+humanos|"
    r"residentes\s+de\s+espa[nñ]a|"
    r"a\s+la\s+atenci[oó]n\s+de:\s+facultad|spam\s+arrest|"
    r"administrador\s+del\s+web|"
    r"ihr\s+postfach|system-administrator|best[aä]tigen\s+sie\s+ihre\s+identit[aä]t|"
    r"gr[oö][sß]enbeschr[aä]nkungen|erneut\s+validieren|"
    r"achtung\s+unsere\s+gesch[aä]tzten|e-mail-konto\s+werden\s+stillgelegt|"
    r"gel[oö]scht\s+aus\s+unserem\s+mail-server|"
    r"postilaatikkoon\s+kiinti[oö]|tallennustila\s+t[aä]yttyy|"
    r"olet\s+napsauttamalla\s+alla\s+olevaa\s+linkki[aä]|"
    r"v[aá]žen[yý]\s+uživateli|e-mailov[eé]ho\s+účtu|"
    r"vaše\s+e-mailov[aá]\s+adresa|spr[aá]vce\s+syst[eé]mu|"
    r"k[ä√]ra\s+kund|k[ćæ]re\s+kunde|kжre\s+kunde|b[äд]ste\s+kund|"
    r"online-tillg[å√]ng|kund\s+tj[ä√]nst|internetbank|"
    r"bank[ö√]verf[ö√]ringar|s[ä√]kerhetsnormer|"
    r"vi\s+beklag[ae]r|vi\s+beklager|kontoen\s+din|"
    r"bekr[äж]fta|s[äд]kra\s+l[äд]nken|telenor\s+webmail"
    r")",
    re.IGNORECASE,
)
ENGLISH_STOPWORDS = {
    "a",
    "about",
    "account",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "click",
    "customer",
    "dear",
    "for",
    "from",
    "have",
    "if",
    "in",
    "information",
    "is",
    "it",
    "link",
    "not",
    "of",
    "on",
    "or",
    "our",
    "please",
    "security",
    "service",
    "that",
    "the",
    "this",
    "to",
    "update",
    "verify",
    "we",
    "will",
    "with",
    "you",
    "your",
}
MEANINGFUL_EDGE_TOKENS = ENGLISH_STOPWORDS | {
    "amazon",
    "america",
    "bank",
    "banking",
    "card",
    "center",
    "chase",
    "citibank",
    "customer",
    "department",
    "ebay",
    "email",
    "fraud",
    "hello",
    "member",
    "notice",
    "online",
    "paypal",
    "regards",
    "sincerely",
    "support",
    "team",
    "thanks",
    "visa",
}


STRICT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def normalize_body_for_modeling(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def strict_token_word_count(value: object) -> int:
    return len(STRICT_TOKEN_RE.findall(normalize_body_for_modeling(value)))


def whitespace_word_count(value: object) -> int:
    return len(normalize_body_for_modeling(value).split())


def passes_body_word_count_filter(value: object) -> bool:
    strict_count = strict_token_word_count(value)
    whitespace_count = whitespace_word_count(value)
    return MIN_BODY_WORDS <= strict_count <= MAX_BODY_WORDS and MIN_BODY_WORDS <= whitespace_count <= MAX_BODY_WORDS


def count_words(value: object) -> int:
    return strict_token_word_count(value)


def normalize_whitespace(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\x00", "").replace("\xa0", " ")

    normalized_lines = []
    previous_blank = False
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False

    value = "\n".join(normalized_lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def decode_html_entities(value: str) -> str:
    previous = value
    for _ in range(3):
        current = html.unescape(previous)
        if current == previous:
            break
        previous = current
    return previous


def strip_html_remnants(value: str) -> str:
    value = HTML_COMMENT_RE.sub(" ", value)
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</(?:p|div|tr|table|li|h[1-6])\s*>", "\n", value)
    value = re.sub(r"(?i)<(?:p|div|tr|table|li|h[1-6])\b[^>]*>", "\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = HTML_COMMENT_DELIMITER_RE.sub(" ", value)
    value = re.sub(
        r"\b(?:href|src|style|class|width|height|border|cellpadding|cellspacing|font-family|bgcolor|align)"
        r"\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return value


def strip_css_declarations(value: str) -> str:
    cleaned_lines = []
    for line in value.splitlines():
        if "{" in line and "}" in line and CSS_PROPERTY_RE.search(line):
            line = CSS_DECLARATION_RE.sub(" ", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def strip_number_markers(value: str) -> str:
    if not NUMBER_MARKER_RE.search(value):
        return value

    value = NUMBER_MARKER_RE.sub("", value)
    paragraphs = re.split(r"\n\s*\n", value)
    value = "\n\n".join(
        re.sub(r"\s*\n\s*", " ", paragraph).strip()
        for paragraph in paragraphs
        if paragraph.strip()
    )
    replacements = {
        "t o": "to",
        "infor m": "inform",
        "th at": "that",
        "Apple I D": "Apple ID",
        "mis sing": "missing",
        "incorr ect": "incorrect",
        "Ope n": "Open",
        "r ead": "read",
        "a ttache d": "attached",
        "verif y": "verify",
        "youra ccount": "your account",
        "inform ation": "information",
        "rece ive": "receive",
        "re sponse": "response",
        "acc ount": "account",
        "wi ll": "will",
        "bl ocked": "blocked",
        "Sincere ly": "Sincerely",
        "su pport": "support",
    }
    for bad, good in replacements.items():
        value = value.replace(bad, good)
    return value


def strip_extraction_remnants(value: str) -> str:
    value = LEADING_TARGET_BLANK_RE.sub("", value)
    value = LEADING_COLON_COMMA_RE.sub("", value)
    value = LEADING_BODY_LABEL_RE.sub("", value)
    value = LEADING_PLUS_EQUALS_RE.sub("", value)
    value = SPACED_DHL_RE.sub("DHL", value)
    value = LEADING_ORPHAN_TAG_CLOSE_RE.sub("", value)
    value = DANGLING_TAG_FRAGMENT_RE.sub("", value)
    value = MALFORMED_FONT_FRAGMENT_RE.sub("", value)
    value = strip_css_declarations(value)
    value = ORPHAN_CSS_STYLE_FRAGMENT_RE.sub("", value)
    value = IMP_FOOTER_RE.sub("", value)
    value = HTML_PLAIN_TEXT_NOTICE_RE.sub("", value)
    value = ANGLE_ARROW_DECORATION_RE.sub(" ", value)
    value = STANDALONE_SEPARATOR_LINE_RE.sub("", value)
    value = STANDALONE_DOT_LINE_RE.sub("", value)
    value = STANDALONE_QUESTION_LINE_RE.sub("", value)
    value = STANDALONE_UNDERSCORE_LINE_RE.sub("", value)
    value = STANDALONE_SMILEY_LINE_RE.sub("", value)
    value = STANDALONE_GT_LINE_RE.sub("", value)
    value = STANDALONE_QUOTE_LINE_RE.sub("", value)
    value = HTML_LINK_REFERENCE_RE.sub("", value)
    value = strip_number_markers(value)
    value = EMPTY_OR_IMAGE_BRACKET_RE.sub("", value)
    previous = value
    for _ in range(5):
        current = LEADING_TITLE_RE.sub("", previous)
        if current == previous:
            break
        previous = current
    return previous


def is_editor_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) > 90:
        return False

    lowered = stripped.lower()
    exact_markers = {
        "clean",
        "false",
        "microsoftinternetexplorer4",
        "normal",
        "user",
    }
    if lowered in exact_markers:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", lowered):
        return True
    if re.fullmatch(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", lowered):
        return True
    if re.fullmatch(r"[\w .@()/*+$'-]{1,35}", stripped) and not re.search(
        r"\b(dear|account|bank|paypal|ebay|amazon|online|security|question|click|update|verify|customer|member)\b",
        lowered,
    ):
        return True
    return False


def strip_editor_metadata_blocks(value: str) -> str:
    lines = value.splitlines()
    remove = [False] * len(lines)

    for index, line in enumerate(lines):
        if line.strip().lower() != "microsoftinternetexplorer4":
            continue

        start = index
        while start > 0 and is_editor_metadata_line(lines[start - 1]):
            start -= 1

        for remove_index in range(start, index + 1):
            remove[remove_index] = True

    return "\n".join(line for index, line in enumerate(lines) if not remove[index])


def decode_quoted_printable_leftovers(value: str) -> str:
    if not QUOTED_PRINTABLE_RE.search(value):
        return value
    decoded = quopri.decodestring(value.encode("utf-8", errors="replace"))
    return decoded.decode("utf-8", errors="replace")


def repair_light_mojibake(value: str) -> str:
    for bad, good in COMMON_MOJIBAKE_REPLACEMENTS.items():
        value = value.replace(bad, good)
    value = value.replace("�", " ")
    return value


def is_random_padding_line(line: str) -> bool:
    original = line.strip()
    cleaned = re.sub(r"[^A-Za-z\s]", " ", line).lower().strip()
    if not cleaned:
        return False

    tokens = re.findall(r"[a-z]+", cleaned)
    if not tokens:
        return False
    if any(token in MEANINGFUL_EDGE_TOKENS and len(token) >= 3 for token in tokens):
        return False

    letters = "".join(tokens)
    if len(letters) < 6:
        return False

    vowel_ratio = sum(1 for char in letters if char in "aeiou") / len(letters)
    long_tokens = sum(1 for token in tokens if len(token) >= 12)
    consonant_clusters = re.findall(r"[bcdfghjklmnpqrstvwxyz]{5,}", letters)

    if len(tokens) == 1:
        return len(letters) >= 6 and vowel_ratio <= 0.34
    if 6 <= len(letters) < 24 and vowel_ratio <= 0.25 and original == original.lower():
        return True
    if len(letters) >= 30 and long_tokens >= 1 and original == original.lower():
        return True
    if len(letters) >= 24 and vowel_ratio <= 0.33:
        return True
    if long_tokens >= 1 and vowel_ratio <= 0.40 and len(consonant_clusters) >= 1:
        return True
    return False


def is_symbol_padding_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    if re.search(r"[A-Za-z0-9]", stripped):
        return False
    symbol_chars = [char for char in stripped if not char.isspace()]
    if len(symbol_chars) < 3:
        return False
    most_common_count = max(symbol_chars.count(char) for char in set(symbol_chars))
    return most_common_count / len(symbol_chars) >= 0.70


def is_hex_padding_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    hex_tokens = re.findall(r"\b0x[0-9a-fA-F]{1,10}\b", stripped)
    if len(hex_tokens) >= 3:
        return True

    non_space_tokens = re.findall(r"\S+", stripped)
    if len(hex_tokens) >= 2 and len(hex_tokens) / max(len(non_space_tokens), 1) >= 0.40:
        return True
    return False


def is_code_padding_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    code_markers = {
        "api",
        "common",
        "create",
        "cvs",
        "dec",
        "end",
        "engine",
        "exe",
        "file",
        "function",
        "hex",
        "include",
        "interface",
        "media",
        "rcs",
        "revision",
        "root",
        "serv",
        "source",
        "stack",
        "start",
        "tmp",
        "type",
        "update",
    }
    tokens = re.findall(r"[A-Za-z0-9$]+", lowered)
    if len(tokens) < 3:
        return False

    marker_count = sum(1 for token in tokens if token in code_markers)
    alnum_gibberish_count = sum(
        1
        for token in tokens
        if re.fullmatch(r"(?=.*[a-z])(?=.*\d)[a-z0-9$]{3,8}", token)
    )
    meaningful_count = sum(1 for token in tokens if token in MEANINGFUL_EDGE_TOKENS)

    if meaningful_count:
        return False
    if marker_count >= 2 and (alnum_gibberish_count >= 1 or len(tokens) <= 8):
        return True
    if marker_count >= 1 and alnum_gibberish_count >= 3:
        return True
    return False


def is_strong_internal_random_padding_line(line: str) -> bool:
    original = line.strip()
    cleaned = re.sub(r"[^A-Za-z\s]", " ", line).lower().strip()
    tokens = re.findall(r"[a-z]+", cleaned)
    if not tokens or any(token in MEANINGFUL_EDGE_TOKENS and len(token) >= 3 for token in tokens):
        return False

    letters = "".join(tokens)
    if len(letters) < 30:
        return False

    vowel_ratio = sum(1 for char in letters if char in "aeiou") / len(letters)
    long_tokens = sum(1 for token in tokens if len(token) >= 12)
    if long_tokens >= 1 and original == original.lower():
        return True
    return len(letters) >= 40 and vowel_ratio <= 0.33


def strip_random_padding_lines(value: str) -> tuple[str, int]:
    lines = value.splitlines()
    remove = [False] * len(lines)

    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    while start < len(lines) and (
        is_random_padding_line(lines[start])
        or is_symbol_padding_line(lines[start])
        or is_hex_padding_line(lines[start])
        or is_code_padding_line(lines[start])
    ):
        remove[start] = True
        start += 1
        while start < len(lines) and not lines[start].strip():
            remove[start] = True
            start += 1

    end = len(lines) - 1
    while end >= 0 and not lines[end].strip():
        end -= 1
    while end >= 0 and (
        is_random_padding_line(lines[end])
        or is_symbol_padding_line(lines[end])
        or is_hex_padding_line(lines[end])
        or is_code_padding_line(lines[end])
    ):
        remove[end] = True
        end -= 1
        while end >= 0 and not lines[end].strip():
            remove[end] = True
            end -= 1

    for index, line in enumerate(lines):
        if not remove[index] and (
            is_strong_internal_random_padding_line(line)
            or is_symbol_padding_line(line)
            or is_hex_padding_line(line)
            or is_code_padding_line(line)
        ):
            remove[index] = True

    removed_count = sum(1 for index, line in enumerate(lines) if remove[index] and line.strip())
    cleaned_lines = [line for index, line in enumerate(lines) if not remove[index]]
    return "\n".join(cleaned_lines), removed_count


def clean_text(value: str | None) -> str:
    value = "" if value is None or pd.isna(value) else str(value)
    value = ZERO_WIDTH_RE.sub("", value)
    value = CONTROL_RE.sub(" ", value)
    value = decode_html_entities(value)
    value = decode_quoted_printable_leftovers(value)
    value = decode_html_entities(value)
    value = strip_html_remnants(value)
    value = strip_extraction_remnants(value)
    value = strip_editor_metadata_blocks(value)
    value = decode_html_entities(value)
    value = repair_light_mojibake(value)
    value = ZERO_WIDTH_RE.sub("", value)
    value = strip_extraction_remnants(value)
    value = strip_editor_metadata_blocks(value)
    value = CONTROL_RE.sub(" ", value)
    return normalize_whitespace(value)


def clean_body_text(value: str | None) -> tuple[str, int]:
    value = clean_text(value)
    value, random_padding_lines_removed = strip_random_padding_lines(value)
    return normalize_whitespace(value), random_padding_lines_removed


def artifact_counts(subject: str, body: str) -> dict:
    text = f"{subject}\n\n{body}"
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    return {
        "replacement_char_count": text.count("�"),
        "mojibake_marker_count": sum(text.count(marker) for marker in MOJIBAKE_PATTERNS),
        "html_tag_count": len(HTML_TAG_RE.findall(text)),
        "html_entity_count": len(HTML_ENTITY_RE.findall(text)),
        "html_attribute_count": len(HTML_ATTRIBUTE_RE.findall(text)),
        "quoted_printable_marker_count": len(QUOTED_PRINTABLE_RE.findall(text)),
        "iso2022jp_escape_count": len(ISO2022JP_ESCAPE_RE.findall(text)),
        "dense_encoded_payload_count": len(DENSE_ENCODED_PAYLOAD_RE.findall(text)),
        "control_char_count": len(CONTROL_RE.findall(text)),
        "non_ascii_count": non_ascii_count,
        "non_ascii_ratio": non_ascii_count / max(len(text), 1),
    }


def is_likely_non_english(subject: str, body: str) -> bool:
    text = f"{subject} {body}"
    if STRONG_NON_ENGLISH_PHRASE_RE.search(text):
        return True

    alpha_chars = [char for char in text if char.isalpha()]
    if len(alpha_chars) < 25:
        return False

    ascii_alpha = sum(1 for char in alpha_chars if char in string.ascii_letters)
    ascii_alpha_ratio = ascii_alpha / len(alpha_chars)
    non_ascii_total_ratio = sum(1 for char in text if ord(char) > 127) / max(len(text), 1)
    non_english_marker_count = len(NON_ENGLISH_MARKER_RE.findall(text))
    words = re.findall(r"[A-Za-z']+", text.lower())
    english_stopword_ratio = sum(1 for word in words if word in ENGLISH_STOPWORDS) / max(len(words), 1)

    if ascii_alpha_ratio < 0.70 or non_ascii_total_ratio >= 0.20:
        return True
    if non_english_marker_count >= 2 and english_stopword_ratio < 0.15:
        return True
    return non_english_marker_count >= 1 and english_stopword_ratio < 0.08


def visible_hash(subject: str, body: str) -> str:
    normalized = f"{subject}\n\n{body}".lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def body_content_hash(body: str) -> str:
    normalized = body.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def build_cleaned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        original_subject = "" if pd.isna(row.get("subject")) else str(row.get("subject"))
        original_body = "" if pd.isna(row.get("body_display")) else str(row.get("body_display"))
        before = artifact_counts(original_subject, original_body)

        clean_subject = clean_text(original_subject)
        clean_body, random_padding_lines_removed = clean_body_text(original_body)
        after = artifact_counts(clean_subject, clean_body)

        cleaned_row = row.to_dict()
        cleaned_row["subject_original"] = original_subject
        cleaned_row["body_display_original"] = original_body
        cleaned_row["subject"] = clean_subject
        cleaned_row["body_display"] = clean_body
        cleaned_row["body_char_count"] = len(clean_body)
        cleaned_row["body_word_count"] = strict_token_word_count(clean_body)
        cleaned_row["body_whitespace_word_count"] = whitespace_word_count(clean_body)
        cleaned_row["random_padding_lines_removed"] = random_padding_lines_removed
        cleaned_row["visible_text_hash_cleaned"] = visible_hash(clean_subject, clean_body)
        cleaned_row["body_content_hash_cleaned"] = body_content_hash(clean_body)
        cleaned_row["is_likely_non_english"] = is_likely_non_english(clean_subject, clean_body)
        cleaned_row["fails_clean_body_length_filter"] = not passes_body_word_count_filter(clean_body)

        for key, value in before.items():
            cleaned_row[f"before_{key}"] = value
        for key, value in after.items():
            cleaned_row[f"after_{key}"] = value

        cleaned_row["has_heavy_encoding_artifact"] = (
            (before["replacement_char_count"] >= 300)
            or (before["mojibake_marker_count"] >= 300)
            or (before["iso2022jp_escape_count"] >= 3)
            or (before["dense_encoded_payload_count"] >= 3)
            or (before["non_ascii_ratio"] >= 0.50)
        )
        cleaned_row["has_literary_padding_artifact"] = bool(
            LITERARY_PADDING_RE.search(f"{original_subject}\n\n{original_body}")
        )
        cleaned_row["has_remaining_text_artifact"] = (
            (after["replacement_char_count"] > 0)
            or (after["mojibake_marker_count"] > 0)
            or (after["html_tag_count"] > 0)
            or (after["html_entity_count"] > 0)
            or (after["html_attribute_count"] > 0)
            or (after["quoted_printable_marker_count"] > 0)
            or (after["control_char_count"] > 0)
        )
        cleaned_row["drop_from_cleaned_phishing"] = (
            cleaned_row["has_heavy_encoding_artifact"]
            or cleaned_row["has_literary_padding_artifact"]
            or cleaned_row["is_likely_non_english"]
            or cleaned_row["fails_clean_body_length_filter"]
            or clean_body == ""
        )
        rows.append(cleaned_row)
    return pd.DataFrame(rows)


def build_summary(flagged: pd.DataFrame, retained_after_drop: pd.DataFrame, final_cleaned: pd.DataFrame) -> pd.DataFrame:
    total = len(flagged)
    rows = [
        {"metric": "input_candidates", "count": total, "percent_of_input": 100.0},
        {
            "metric": "dropped_heavy_encoding_or_corruption",
            "count": int(flagged["has_heavy_encoding_artifact"].sum()),
            "percent_of_input": round(flagged["has_heavy_encoding_artifact"].sum() / total * 100, 2),
        },
        {
            "metric": "dropped_literary_or_non_phishing_padding",
            "count": int(flagged["has_literary_padding_artifact"].sum()),
            "percent_of_input": round(flagged["has_literary_padding_artifact"].sum() / total * 100, 2),
        },
        {
            "metric": "dropped_likely_non_english",
            "count": int(flagged["is_likely_non_english"].sum()),
            "percent_of_input": round(flagged["is_likely_non_english"].sum() / total * 100, 2),
        },
        {
            "metric": "dropped_failed_clean_body_length",
            "count": int(flagged["fails_clean_body_length_filter"].sum()),
            "percent_of_input": round(flagged["fails_clean_body_length_filter"].sum() / total * 100, 2),
        },
        {
            "metric": "total_dropped_after_cleaning",
            "count": int(flagged["drop_from_cleaned_phishing"].sum()),
            "percent_of_input": round(flagged["drop_from_cleaned_phishing"].sum() / total * 100, 2),
        },
        {
            "metric": "retained_after_artifact_language_length_rules",
            "count": len(retained_after_drop),
            "percent_of_input": round(len(retained_after_drop) / total * 100, 2),
        },
        {
            "metric": "rows_with_remaining_artifact_after_cleaning",
            "count": int(final_cleaned["has_remaining_text_artifact"].sum()),
            "percent_of_input": round(final_cleaned["has_remaining_text_artifact"].sum() / total * 100, 2),
        },
        {
            "metric": "rows_with_random_padding_removed",
            "count": int((flagged["random_padding_lines_removed"] > 0).sum()),
            "percent_of_input": round((flagged["random_padding_lines_removed"] > 0).sum() / total * 100, 2),
        },
        {
            "metric": "random_padding_lines_removed",
            "count": int(flagged["random_padding_lines_removed"].sum()),
            "percent_of_input": "",
        },
        {
            "metric": "remaining_html_tag_rows",
            "count": int((final_cleaned["after_html_tag_count"] > 0).sum()),
            "percent_of_input": round((final_cleaned["after_html_tag_count"] > 0).sum() / total * 100, 2),
        },
        {
            "metric": "remaining_html_entity_rows",
            "count": int((final_cleaned["after_html_entity_count"] > 0).sum()),
            "percent_of_input": round((final_cleaned["after_html_entity_count"] > 0).sum() / total * 100, 2),
        },
        {
            "metric": "remaining_quoted_printable_rows",
            "count": int((final_cleaned["after_quoted_printable_marker_count"] > 0).sum()),
            "percent_of_input": round((final_cleaned["after_quoted_printable_marker_count"] > 0).sum() / total * 100, 2),
        },
        {
            "metric": "remaining_replacement_char_rows",
            "count": int((final_cleaned["after_replacement_char_count"] > 0).sum()),
            "percent_of_input": round((final_cleaned["after_replacement_char_count"] > 0).sum() / total * 100, 2),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Nazario standalone pre-2020 phishing candidates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "processed" / "Nazario_Prelim_Standalone_Pre2020_Candidates.csv",
    )
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "processed")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    flagged = build_cleaned_dataset(df)
    flagged["is_duplicate_visible_text_after_cleaning"] = False
    flagged["is_duplicate_body_content_after_cleaning"] = False
    flagged["is_duplicate_after_cleaning"] = False

    retained_after_drop = flagged.loc[~flagged["drop_from_cleaned_phishing"]].copy()
    retained_after_drop["is_duplicate_visible_text_after_cleaning"] = retained_after_drop.duplicated(
        "visible_text_hash_cleaned",
        keep="first",
    )
    retained_after_drop["is_duplicate_body_content_after_cleaning"] = retained_after_drop.duplicated(
        "body_content_hash_cleaned",
        keep="first",
    )
    retained_after_drop["is_duplicate_after_cleaning"] = (
        retained_after_drop["is_duplicate_visible_text_after_cleaning"]
        | retained_after_drop["is_duplicate_body_content_after_cleaning"]
    )
    duplicate_columns = [
        "is_duplicate_visible_text_after_cleaning",
        "is_duplicate_body_content_after_cleaning",
        "is_duplicate_after_cleaning",
    ]
    flagged.loc[retained_after_drop.index, duplicate_columns] = retained_after_drop[duplicate_columns]
    cleaned = retained_after_drop.loc[~retained_after_drop["is_duplicate_after_cleaning"]].copy()

    dropped = flagged.loc[flagged["drop_from_cleaned_phishing"]].copy()
    duplicate_review = retained_after_drop.loc[retained_after_drop["is_duplicate_after_cleaning"]].copy()
    summary = build_summary(flagged, retained_after_drop, cleaned)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "metric": "duplicates_removed_after_cleaning",
                        "count": int(retained_after_drop["is_duplicate_after_cleaning"].sum()),
                        "percent_of_input": round(
                            retained_after_drop["is_duplicate_after_cleaning"].sum() / len(flagged) * 100,
                            2,
                        ),
                    },
                    {
                        "metric": "subject_body_duplicates_removed_after_cleaning",
                        "count": int(retained_after_drop["is_duplicate_visible_text_after_cleaning"].sum()),
                        "percent_of_input": round(
                            retained_after_drop["is_duplicate_visible_text_after_cleaning"].sum() / len(flagged) * 100,
                            2,
                        ),
                    },
                    {
                        "metric": "body_content_duplicates_removed_after_cleaning",
                        "count": int(retained_after_drop["is_duplicate_body_content_after_cleaning"].sum()),
                        "percent_of_input": round(
                            retained_after_drop["is_duplicate_body_content_after_cleaning"].sum() / len(flagged) * 100,
                            2,
                        ),
                    },
                    {
                        "metric": "final_cleaned_candidates",
                        "count": len(cleaned),
                        "percent_of_input": round(len(cleaned) / len(flagged) * 100, 2),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    cleaned.to_csv(args.outdir / "Nazario_Cleaned_Standalone_Pre2020_Candidates.csv", index=False)
    dropped.to_csv(args.outdir / "Nazario_Dropped_Cleaning_Review.csv", index=False)
    duplicate_review.to_csv(args.outdir / "Nazario_Duplicate_Content_Review.csv", index=False)
    flagged.to_csv(args.outdir / "Nazario_Cleaning_Audit_All_Candidates.csv", index=False)
    summary.to_csv(args.outdir / "Nazario_Cleaning_Transformation_Summary.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(f"Saved cleaned candidates: {args.outdir / 'Nazario_Cleaned_Standalone_Pre2020_Candidates.csv'}")
    print(f"Saved dropped review: {args.outdir / 'Nazario_Dropped_Cleaning_Review.csv'}")


if __name__ == "__main__":
    main()
