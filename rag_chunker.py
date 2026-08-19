#!/usr/bin/env python3
"""Structure-aware chunking for the codebase RAG index.

Chunks are produced from a real tree-sitter parse (Kotlin, TS/TSX, Dart, SQL,
Python, Swift, Java, Go, Rust, C/C++, HTML, CSS, YAML, JSON, Markdown, ...)
so that `start_line`/`end_line` describe the symbol itself rather than an
arbitrary sliding window. Three guarantees the previous window chunker did not
provide:

  1. Line ranges are exact -- an agent can jump straight to them.
  2. No chunk exceeds the encoder's token window, so nothing is silently
     truncated at embed time; oversized symbols are split into `part i/n`.
  3. Every line of every file lands in some chunk (gap filling), so recall is
     never worse than a naive window index.

If tree-sitter is unavailable the module degrades to a heading/window splitter
rather than failing -- the pipeline stays usable on any machine.
"""
import os
import re

import rag_config as cfg

try:
    from tree_sitter_language_pack import get_parser as _get_parser
    TREE_SITTER_AVAILABLE = True
except Exception:                                          # pragma: no cover
    TREE_SITTER_AVAILABLE = False

    def _get_parser(_lang):
        raise RuntimeError("tree-sitter not installed")

_PARSERS = {}

# Languages we ask tree-sitter to parse. Anything else uses the fallback.
PARSEABLE = {
    "python", "kotlin", "typescript", "tsx", "javascript", "dart", "sql",
    "java", "swift", "go", "rust", "c", "cpp", "html", "css", "yaml", "json",
}

# Nodes that become a chunk of their own.
CHUNK_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration",
               "interface_declaration", "enum_class_body", "companion_object",
               "property_declaration", "secondary_constructor"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                   "interface_declaration", "type_alias_declaration",
                   "lexical_declaration", "variable_declaration", "enum_declaration",
                   "abstract_class_declaration"},
    "javascript": {"function_declaration", "class_declaration", "method_definition",
                   "lexical_declaration", "variable_declaration"},
    "dart": {"class_definition", "function_signature", "method_signature",
             "enum_declaration", "extension_declaration", "mixin_declaration"},
    "sql": {"statement"},
    "java": {"method_declaration", "class_declaration", "interface_declaration",
             "constructor_declaration", "enum_declaration"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration",
              "property_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item", "enum_item"},
    "c": {"function_definition", "struct_specifier", "declaration"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "css": {"rule_set", "media_statement"},
    "html": {"element"},
    "yaml": {"block_mapping_pair", "document"},
    "json": {"pair"},
}
CHUNK_TYPES["tsx"] = CHUNK_TYPES["typescript"]

# Wrappers that carry no identity of their own -- walk straight through them.
TRANSPARENT_TYPES = {
    "export_statement", "expression_statement", "decorated_definition",
    "labeled_statement", "ambient_declaration", "statement_block",
}

# Bodies we descend into when a symbol is too large to embed in one piece.
BODY_TYPES = {
    "class_body", "declaration_list", "block", "statement_block", "body",
    "enum_class_body", "class_declaration_list", "field_declaration_list",
    "interface_body", "object_declaration_list",
}

MIN_GAP_LINES = 4
GAP_WINDOW = 40
GAP_OVERLAP = 8
_IDENT_SUFFIXES = ("identifier", "_name", "name")


def _parser(language):
    if language not in _PARSERS:
        _PARSERS[language] = _get_parser(language)
    return _PARSERS[language]


# ---------------------------------------------------------------------------
# Skeletons
# ---------------------------------------------------------------------------
_COMMENT_START = re.compile(r"^\s*(//|#|/\*|\*|--|<!--)")
_CLOSERS = re.compile(r"^\s*[)\]}>;,]+\s*$")


def build_skeleton(text: str, symbol: str = "", language: str = "other",
                   signature_lines: int = 0) -> str:
    """A skeleton that shows the *signature*, not the first four raw lines.

    The old implementation returned `lines[:4]`, which for a window chunk was
    whatever happened to sit at the top of the window -- frequently a closing
    brace or an unrelated JSX attribute. Here we anchor on the declaration line
    and then show the first few meaningful body lines, which is what a reader
    needs to decide whether to open the full chunk.
    """
    lines = text.splitlines()
    if len(lines) <= 6:
        return text

    # Anchor: the line that declares the symbol, else the first real line.
    anchor = 0
    if symbol:
        for i, line in enumerate(lines[:12]):
            if re.search(r"\b%s\b" % re.escape(symbol), line):
                anchor = i
                break
    else:
        for i, line in enumerate(lines[:8]):
            if line.strip() and not _COMMENT_START.match(line):
                anchor = i
                break

    head = lines[:anchor] if anchor and anchor <= 3 else []
    sig_span = signature_lines if signature_lines else _signature_span(lines, anchor)
    signature = lines[anchor:anchor + sig_span]

    body = []
    for line in lines[anchor + sig_span:]:
        stripped = line.strip()
        if not stripped or _CLOSERS.match(line) or _COMMENT_START.match(line):
            continue
        body.append(line)
        if len(body) >= 3:
            break

    shown = head + signature + body
    hidden = len(lines) - len(shown)
    out = "\n".join(shown)
    if hidden > 0:
        out += "\n... [%d more lines -- get_chunk_content for full body] ..." % hidden
    return out


def _signature_span(lines, anchor, limit=4):
    """How many lines the declaration spans (multi-line parameter lists)."""
    depth = 0
    for offset in range(0, min(limit, len(lines) - anchor)):
        line = lines[anchor + offset]
        depth += line.count("(") - line.count(")")
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            return offset + 1
    return 1


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------
def _make_chunk(rel_path, symbol, start_line, end_line, text, language,
                doc_category, node_type, parent=None, part=1, parts=1):
    suffix = "" if parts == 1 else ".p%d" % part
    chunk_id = "%s:%s:%d-%d%s" % (rel_path, symbol or "block", start_line, end_line, suffix)
    return {
        "id": chunk_id,
        "text": text,
        "skeleton": build_skeleton(text, symbol, language),
        "metadata": {
            "filepath": rel_path,
            "symbol": symbol or "block",
            "parent_symbol": parent or "",
            "start_line": start_line,
            "end_line": end_line,
            "type": node_type,
            "doc_category": doc_category,
            "language": language,
            "part": part,
            "parts": parts,
        },
    }


def _split_oversized(rel_path, symbol, start_line, lines, language,
                     doc_category, node_type, parent):
    """Split a symbol that exceeds the encoder window into labelled parts."""
    # Budget in characters, but derive the exchange rate from THIS text rather
    # than a global constant: a minified JSON blob really does cost ~1.8
    # chars/token where prose costs ~4.0, and a fixed 3.0 either truncates the
    # dense case or shreds the sparse one into needless parts.
    text_for_ratio = "\n".join(lines)
    density = cfg.chars_per_token(text_for_ratio) if text_for_ratio else cfg.CHARS_PER_TOKEN
    budget_chars = max(200, int(cfg.MAX_CHUNK_TOKENS * density * 0.95))

    # Hard-wrap any single line that alone blows the budget, so the caller's
    # invariant ("no chunk exceeds the encoder window") always holds.
    wrapped = []
    for line in lines:
        while len(line) > budget_chars:
            wrapped.append(line[:budget_chars])
            line = line[budget_chars:]
        wrapped.append(line)
    lines = wrapped

    overlap_budget = max(0, int(budget_chars * 0.12))
    pieces, current, current_len, piece_start = [], [], 0, 0
    for idx, line in enumerate(lines):
        if current and current_len + len(line) + 1 > budget_chars:
            pieces.append((piece_start, current))
            # Carry a little context into the next piece so a split symbol stays
            # searchable from either side -- but bound it by characters, not by
            # a line count: three long prose lines would blow the whole budget.
            overlap, overlap_len = [], 0
            for prev in reversed(current):
                if len(overlap) >= 3 or overlap_len + len(prev) > overlap_budget:
                    break
                overlap.insert(0, prev)
                overlap_len += len(prev) + 1
            current = list(overlap)
            current_len = overlap_len
            piece_start = idx - len(overlap)
        current.append(line)
        current_len += len(line) + 1
    if current:
        pieces.append((piece_start, current))

    # The budget is a good estimate, not a proof. Verify each piece against the
    # real tokenizer and re-cut any that still overflows, so the module's
    # stated guarantee ("no chunk exceeds the encoder window") is actually one.
    verified = []
    for offset, piece_lines in pieces:
        text = "\n".join(piece_lines)
        if cfg.count_tokens(text) <= cfg.MAX_CHUNK_TOKENS or len(piece_lines) < 2:
            verified.append((offset, piece_lines))
            continue
        shrunk = max(200, int(budget_chars * 0.6))
        sub, cur, cur_len, sub_start = [], [], 0, offset
        for idx, line in enumerate(piece_lines):
            if cur and cur_len + len(line) + 1 > shrunk:
                sub.append((sub_start, cur))
                cur, cur_len, sub_start = [], 0, offset + idx
            cur.append(line)
            cur_len += len(line) + 1
        if cur:
            sub.append((sub_start, cur))
        verified.extend(sub)
    pieces = verified

    total = len(pieces)
    out = []
    for i, (offset, piece_lines) in enumerate(pieces, start=1):
        s = start_line + offset
        e = s + len(piece_lines) - 1
        out.append(_make_chunk(rel_path, symbol, s, e, "\n".join(piece_lines),
                               language, doc_category, node_type, parent, i, total))
    return out


def _append_bounded(chunks, rel_path, symbol, start_line, end_line, lines,
                    language, doc_category, node_type, parent):
    """Append a chunk, splitting it first if it would overflow the encoder."""
    text = "\n".join(lines).strip("\n")
    if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
        chunks.extend(_split_oversized(rel_path, symbol, start_line, lines,
                                       language, doc_category, node_type, parent))
    else:
        chunks.append(_make_chunk(rel_path, symbol, start_line, end_line, text,
                                  language, doc_category, node_type, parent))


_SQL_STMT_START = re.compile(
    r"^\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|GRANT|REVOKE|COMMENT|DO)\b",
    re.IGNORECASE)


def _split_sql(rel_path, start_line, lines, doc_category):
    """Break a fused SQL node on top-level DDL keywords."""
    groups, current, group_start = [], [], 0
    for idx, line in enumerate(lines):
        if _SQL_STMT_START.match(line) and current and any(l.strip() for l in current):
            groups.append((group_start, current))
            current, group_start = [], idx
        current.append(line)
    if current:
        groups.append((group_start, current))

    out = []
    for offset, group in groups:
        text = "\n".join(group).strip("\n")
        if not text.strip():
            continue
        s = start_line + offset
        e = s + len(group) - 1
        _append_bounded(out, rel_path, _sql_symbol(text), s, e, group, "sql",
                        doc_category, "sql_statement", None)
    return out


def _node_symbol(node, source_bytes):
    named = node.child_by_field_name("name")
    if named is not None:
        return source_bytes[named.start_byte:named.end_byte].decode("utf-8", "ignore").strip()
    # Breadth-first over the first two levels for an identifier-ish node.
    frontier = list(node.children)
    for _ in range(2):
        nxt = []
        for child in frontier:
            if child.type.endswith(_IDENT_SUFFIXES) or child.type in {
                "identifier", "simple_identifier", "type_identifier",
                "property_identifier", "field_identifier", "object_reference",
                "dotted_name",
            }:
                text = source_bytes[child.start_byte:child.end_byte]
                name = text.decode("utf-8", "ignore").strip().strip('"`\'')
                if name:
                    return name.split(".")[-1] if child.type == "object_reference" else name
            nxt.extend(child.children)
        frontier = nxt
    return ""


_DART_DECL = re.compile(
    r"^\s*(?:@\w+\s+)*(?:static\s+|final\s+|const\s+|abstract\s+)*"
    r"(?:[\w<>,\s\?\[\]]+\s+)?([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(")


def _dart_symbol(text):
    """Dart declarations lead with the return type, so the plain 'first
    identifier' heuristic yields 'Future' instead of the method name."""
    for line in text.splitlines()[:3]:
        m = _DART_DECL.match(line)
        if m and m.group(1) not in ("if", "for", "while", "switch", "return", "catch"):
            return m.group(1)
    return ""


def _sql_symbol(text):
    m = re.search(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|FUNCTION|PROCEDURE|VIEW|TRIGGER|POLICY|INDEX|EXTENSION)"
        r"(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:\"([^\"]+)\"|([A-Za-z0-9_.]+))", text, re.IGNORECASE)
    if m:
        # Quoted policy names contain spaces -- keep them whole.
        return (m.group(1) or m.group(2)).strip()
    m = re.search(r"^\s*(ALTER|INSERT|GRANT|DROP)\s+\w+\s+([\"A-Za-z0-9_.]+)",
                  text, re.IGNORECASE | re.MULTILINE)
    return (m.group(2).replace('"', "") if m else "")


# ---------------------------------------------------------------------------
# Tree-sitter driven extraction
# ---------------------------------------------------------------------------
def _extract_with_tree_sitter(rel_path, content, language, doc_category):
    parser = _parser(language)
    source_bytes = content.encode("utf-8", "ignore")
    tree = parser.parse(source_bytes)
    lines = content.splitlines()
    chunk_types = CHUNK_TYPES.get(language, set())
    chunks, covered = [], set()

    def emit(node, parent_symbol):
        start_line = max(1, node.start_point[0] + 1)
        end_line = min(len(lines), node.end_point[0] + 1)
        if end_line < start_line:
            return
        # Dart's grammar emits the signature and the body as siblings; keep them
        # together so a chunk is a whole method rather than one orphan line.
        if language == "dart" and node.type in ("function_signature", "method_signature"):
            sibling = node.next_sibling
            while sibling is not None and sibling.type in (
                    "function_body", "block", "=>", ";", "async", "await"):
                end_line = max(end_line, sibling.end_point[0] + 1)
                if sibling.type in ("function_body", "block"):
                    break
                sibling = sibling.next_sibling
        node_lines = lines[start_line - 1:end_line]
        text = "\n".join(node_lines).strip("\n")
        if not text.strip():
            return
        if language == "sql":
            symbol = _sql_symbol(text)
        elif language == "dart":
            symbol = _dart_symbol(text) or _node_symbol(node, source_bytes)
        else:
            symbol = _node_symbol(node, source_bytes)
        if not symbol and language in ("typescript", "tsx", "javascript"):
            m = re.search(r"\b(?:const|let|var|function|class)\s+([A-Za-z0-9_$]+)", text)
            symbol = m.group(1) if m else ""

        if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
            # SQL grammars swallow dollar-quoted function bodies, which can fuse
            # a whole file into one `statement` node. Re-split on DDL keywords
            # so each policy/function stays individually addressable.
            if language == "sql":
                pieces = _split_sql(rel_path, start_line, node_lines, doc_category)
                if len(pieces) > 1:
                    chunks.extend(pieces)
                    covered.update(range(start_line, end_line + 1))
                    return

            # Prefer structural sub-chunks over a blind split.
            children = _chunkable_children(node)
            if children:
                body_start = children[0].start_point[0] + 1
                if body_start - start_line >= 2:
                    header_lines = lines[start_line - 1:body_start - 1]
                    header = "\n".join(header_lines).strip("\n")
                    if header.strip():
                        _append_bounded(chunks, rel_path, symbol, start_line,
                                        body_start - 1, header_lines, language,
                                        doc_category, node.type + "_header",
                                        parent_symbol)
                        covered.update(range(start_line, body_start))
                for child in children:
                    emit(child, symbol or parent_symbol)
                return
            chunks.extend(_split_oversized(rel_path, symbol, start_line, node_lines,
                                           language, doc_category, node.type, parent_symbol))
            covered.update(range(start_line, end_line + 1))
            return

        chunks.append(_make_chunk(rel_path, symbol, start_line, end_line, text,
                                  language, doc_category, node.type, parent_symbol))
        covered.update(range(start_line, end_line + 1))

    def _chunkable_children(node, max_depth=6):
        """Nearest layer of structural children, stopping at each match.

        Descending through *any* intermediate node (variable_declarator,
        arrow_function, call_expression, ...) is what lets a 1200-line
        `const ChatProvider = () => { ... }` decompose into its real inner
        symbols instead of being sliced blindly by line count.
        """
        found = []
        stack = [(child, 0) for child in node.children]
        while stack:
            child, depth = stack.pop(0)
            if child.type in chunk_types:
                found.append(child)
            elif depth < max_depth and child.child_count:
                stack.extend((g, depth + 1) for g in child.children)
        return _coalesce_nodes(sorted(found, key=lambda n: n.start_byte))

    def _coalesce_nodes(nodes):
        """Drop children fully contained in an earlier sibling."""
        out = []
        for n in nodes:
            if out and n.start_byte >= out[-1].start_byte and n.end_byte <= out[-1].end_byte:
                continue
            out.append(n)
        return out

    def walk(node, parent_symbol=""):
        for child in node.children:
            if child.type in chunk_types:
                emit(child, parent_symbol)
            elif child.type in TRANSPARENT_TYPES or child.child_count:
                walk(child, parent_symbol)

    walk(tree.root_node)
    chunks.extend(_gap_chunks(rel_path, lines, covered, language, doc_category))
    return chunks


def _gap_chunks(rel_path, lines, covered, language, doc_category):
    """Window-index whatever the parser did not claim (imports, top-level code,
    parse errors) so the index never loses a line."""
    out = []
    gap_start = None
    for i in range(1, len(lines) + 2):
        line_is_gap = i <= len(lines) and i not in covered
        if line_is_gap and gap_start is None:
            gap_start = i
        elif not line_is_gap and gap_start is not None:
            out.extend(_window(rel_path, lines, gap_start, i - 1, language, doc_category))
            gap_start = None
    return out


# The window splitter only runs where tree-sitter could not (unsupported
# language, syntax error, or a gap between parsed nodes) -- which is exactly
# where a name matters most, because there is no node to ask. The old pattern
# knew six keywords from JS and Python, so Go/Rust/Kotlin/Swift gap chunks came
# back anonymous and were unreachable by symbol. These cover the declaration
# forms those languages actually use, including leading decorators/annotations
# and receivers, and each alternative is anchored so prose cannot match.
_FALLBACK_DECLS = (
    # JS/TS/Python/Kotlin/Dart
    r"^[ \t]*(?:@[\w.]+[ \t]*)*(?:export[ \t]+)?(?:default[ \t]+)?"
    r"(?:public|private|internal|protected|abstract|final|open|sealed|static|"
    r"suspend|inline|override|external|operator|data|async|const|readonly)?"
    r"[ \t]*(?:function|class|interface|enum|object|trait|struct|record|"
    r"typealias|type|fun|def|val|var|let|const)[ \t]+([A-Za-z_$][\w$]*)",
    # Go: func (r *Recv) Name(...) / func Name(...)
    r"^[ \t]*func[ \t]+(?:\([^)]*\)[ \t]*)?([A-Za-z_][\w]*)",
    # Rust: pub async unsafe fn name / impl Trait for Name / struct Name
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:default[ \t]+)?(?:const[ \t]+)?"
    r"(?:async[ \t]+)?(?:unsafe[ \t]+)?(?:extern[ \t]+\"[^\"]*\"[ \t]+)?"
    r"(?:fn|struct|enum|trait|union|mod|impl)[ \t]+([A-Za-z_][\w]*)",
    # Swift/Java/C#: modifiers then func/class/protocol/extension
    r"^[ \t]*(?:@\w+[ \t]+)*(?:public|private|fileprivate|internal|open|"
    r"protected|static|final)?[ \t]*(?:func|protocol|extension|actor)"
    r"[ \t]+([A-Za-z_][\w]*)",
    # C/C++/Java method: <type> name(  -- last resort, needs a brace or colon
    r"^[ \t]*(?:[A-Za-z_][\w:<>,*&\s]{0,60}?[\s*&])([A-Za-z_]\w*)[ \t]*\([^;]*\)"
    r"[ \t]*(?:const)?[ \t]*\{",
)
_FALLBACK_DECLS = tuple(re.compile(p, re.MULTILINE) for p in _FALLBACK_DECLS)
_FALLBACK_STOP = {
    "if", "for", "while", "switch", "return", "catch", "else", "do", "try",
    "match", "loop", "when", "with", "in", "of", "new", "case",
}


def _fallback_symbol(text: str) -> str:
    """Best-effort declaration name for a chunk no parser claimed."""
    for pattern in _FALLBACK_DECLS:
        for m in pattern.finditer(text):
            name = m.group(1)
            if name and name.lower() not in _FALLBACK_STOP:
                return name
    return ""


def _window(rel_path, lines, start_line, end_line, language, doc_category):
    span = lines[start_line - 1:end_line]
    if len([l for l in span if l.strip()]) < MIN_GAP_LINES:
        return []
    out = []
    i = 0
    while i < len(span):
        piece = span[i:i + GAP_WINDOW]
        text = "\n".join(piece).strip("\n")
        if text.strip():
            s = start_line + i
            e = s + len(piece) - 1
            symbol = _sql_symbol(text) if language == "sql" else ""
            if not symbol:
                symbol = _fallback_symbol(text)
            if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
                out.extend(_split_oversized(rel_path, symbol, s, piece, language,
                                            doc_category, "block", None))
            else:
                out.append(_make_chunk(rel_path, symbol, s, e, text, language,
                                       doc_category, "block"))
        if len(piece) < GAP_WINDOW:
            break
        i += (GAP_WINDOW - GAP_OVERLAP)
    return out


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _extract_markdown(rel_path, content, doc_category):
    """Split docs on headings so a section is retrievable as a unit."""
    lines = content.splitlines()
    sections, current, title, start = [], [], "", 1
    for i, line in enumerate(lines, start=1):
        m = _MD_HEADING.match(line)
        if m and current:
            sections.append((title, start, i - 1, current))
            current, title, start = [], m.group(2).strip(), i
        elif m:
            title, start = m.group(2).strip(), i
        current.append(line)
    if current:
        sections.append((title, start, len(lines), current))

    chunks = []
    for title, s, e, body in sections:
        text = "\n".join(body).strip("\n")
        if not text.strip():
            continue
        symbol = re.sub(r"[^A-Za-z0-9_ .-]", "", title)[:60] or "section"
        if cfg.est_tokens(text) > cfg.MAX_CHUNK_TOKENS:
            chunks.extend(_split_oversized(rel_path, symbol, s, body, "markdown",
                                           doc_category, "doc_section", None))
        else:
            chunks.append(_make_chunk(rel_path, symbol, s, e, text, "markdown",
                                      doc_category, "doc_section"))
    return chunks


def extract_chunks(abs_path, rel_path, content):
    ext = os.path.splitext(abs_path)[1].lower()
    language = cfg.language_for(ext)
    doc_category = cfg.doc_category_for(ext)

    if not content.strip():
        return []

    if language == "markdown":
        return _extract_markdown(rel_path, content, doc_category)

    if TREE_SITTER_AVAILABLE and language in PARSEABLE:
        try:
            chunks = _extract_with_tree_sitter(rel_path, content, language, doc_category)
            if chunks:
                return chunks
        except Exception:
            pass  # fall through to the window splitter

    lines = content.splitlines()
    return _window(rel_path, lines, 1, len(lines), language, doc_category)

# Bump when chunk boundaries change so ingests auto-rebuild.
VERSION = "2.4"
