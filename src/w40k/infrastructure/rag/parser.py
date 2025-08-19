from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PageElement, Tag

PARSER_VERSION = "w40k-parser/1.0.1"

_SPACE = re.compile(r"\s+")
_COMMA_SPACES = re.compile(r"\s*,\s*")  # normalize comma spacing
_MULTI_COMMAS = re.compile(r"(?:,\s*){2,}")  # collapse repeated commas
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
_MULTI_SPACE = re.compile(r"\s{2,}")
_SPACE_AFTER_OPENQUOTE = re.compile(
    r'([“"\'(\[]) +'
)  # space after opening quote/bracket
_SPACE_BEFORE_CLOSEQUOTE = re.compile(
    r' +([”"\')\]])'
)  # space before closing quote/bracket


def _clean_commas(s: str) -> str:
    if not s:
        return s
    # normalize spaces around commas
    s = _COMMA_SPACES.sub(", ", s)
    # collapse ", , ," -> ", "
    s = _MULTI_COMMAS.sub(", ", s)
    # remove spaces before punctuation like " ,", " ."
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    # collapse leftover multiple spaces
    s = _MULTI_SPACE.sub(" ", s)
    # trim leading/trailing commas and spaces
    return s.strip(" ,;")


def _tidy_punct(s: str) -> str:
    if not s:
        return s
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _SPACE_AFTER_OPENQUOTE.sub(r"\1", s)
    s = _SPACE_BEFORE_CLOSEQUOTE.sub(r"\1", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


def _normspace(s: Optional[str]) -> str:
    return _SPACE.sub(" ", (s or "").strip())


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _drop_ref_sups(node: PageElement) -> None:
    # Remove MediaWiki-style citation/footnote superscripts
    for sup in getattr(node, "select", lambda *_, **__: [])(
        "sup.reference, sup[role='doc-noteref'], sup[class*='reference'], sup.ref"
    ):
        # sup is a Tag here
        sup.decompose()


def _text_of(node: Optional[PageElement]) -> str:
    if node is None:
        return ""
    _drop_ref_sups(node)
    get_text = getattr(node, "get_text", None)
    if callable(get_text):
        return _tidy_punct(_normspace(get_text(" ", strip=True))) # type: ignore
    return ""


def _canonical_url(soup: BeautifulSoup) -> Optional[str]:
    link = soup.find("link", rel="canonical")
    if link is None:
        return None
    href = link.get("href") if isinstance(link, Tag) else None
    return href if isinstance(href, str) else None


def _meta_content(soup: BeautifulSoup, key: str, *, by: str = "name") -> Optional[str]:
    el: Optional[Tag]
    if by == "name":
        el = soup.find("meta", attrs={"name": key})  # type: ignore[assignment]
    else:
        el = soup.find("meta", attrs={"property": key})  # type: ignore[assignment]
    if el is None:
        return None
    content = el.get("content")
    return content if isinstance(content, str) else None


def _first_heading(soup: BeautifulSoup) -> Optional[str]:
    h = soup.select_one("#firstHeading, h1.page-header__title, h1")
    return _text_of(h) if h is not None else None


def _content_root(soup: BeautifulSoup) -> Optional[PageElement]:
    # Prefer MediaWiki parser output inside the content area
    return (
        soup.select_one("#mw-content-text .mw-parser-output")
        or soup.select_one("article .mw-parser-output")
        or soup.select_one("#mw-content-text")  # fallback
        or soup.select_one("article")
    )


class Preface(TypedDict, total=False):
    paragraphs: list[str]
    lists: list[dict]
    tables: list[dict]
    figures: list[dict]
    quotes: list[str]


def _extract_preface(content: Optional[PageElement]) -> Preface:
    """
    Collect everything before the first h2/h3 as the 'preface':
    paragraphs, lists, tables, figures, quotes.
    Infobox/TOC/hatnotes are skipped via ancestor check.
    """
    preface: Preface = {
        "paragraphs": [],
        "lists": [],
        "tables": [],
        "figures": [],
        "quotes": [],
    }
    if content is None:
        return preface

    for child in getattr(content, "children", []):
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, (Tag, PageElement)):
            continue

        # Stop at the first real section
        if _is_section_heading(child):
            break

        # Safety: skip anything that lives under infobox/TOC/hatnotes containers
        if isinstance(child, Tag) and _has_forbidden_ancestor(child):
            continue

        name = getattr(child, "name", None)
        if name == "p":
            txt = _text_of(child)
            if txt:
                preface["paragraphs"].append(txt)
        elif name in ("ul", "ol"):
            preface["lists"].append(
                {
                    "ordered": (name == "ol"),
                    "items": [
                        {"text": _text_of(li), "children": _extract_lists(li)}
                        for li in getattr(child, "find_all", lambda *_, **__: [])("li", recursive=False)
                    ],
                }
            )
        elif name == "table":
            preface["tables"].append(_extract_table(child))  # type: ignore[arg-type]
        elif name == "blockquote":
            preface["quotes"].append(_text_of(child))
        elif name == "figure":
            img = getattr(child, "find", lambda *_, **__: None)("img")
            src = getattr(img, "get", lambda *_, **__: None)("src") if img is not None else None
            cap = getattr(child, "find", lambda *_, **__: None)("figcaption")
            if isinstance(src, str):
                preface["figures"].append({"src": src, "caption": _text_of(cap)})
        else:
            # ignore other node types
            pass

    return preface


def _extract_categories(soup: BeautifulSoup) -> List[str]:
    cats: List[str] = []
    for a in soup.select('#catlinks a[href*="/Category:"]'):
        txt = _text_of(a).replace("Category:", "").strip()
        if txt:
            cats.append(txt)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: List[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _value_text(cell: Optional[PageElement]) -> str:
    """
    Normalize an infobox value cell:
    - remove citation superscripts
    - if it contains <li>, join items with ', '
    - else, use get_text with separator=', ' to respect <br>
    """
    if cell is None:
        return ""
    _drop_ref_sups(cell)

    find_all = getattr(cell, "find_all", None)
    if callable(find_all):
        lis = find_all("li", recursive=True)
        if lis:
            parts = [
                _normspace(
                    getattr(li, "get_text", lambda *_, **__: "")(" ", strip=True)
                )
                for li in lis  # type: ignore[misc]
            ]
            return _clean_commas(", ".join(p for p in parts if p))

        get_text = getattr(cell, "get_text", None)
        if callable(get_text):
            # Respect <br> boundaries, then clean commas/spaces
            return _clean_commas(_normspace(get_text(", ", strip=True)))  # type: ignore[arg-type]

    return ""


def _value_links(cell: Optional[PageElement]) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    if cell is None:
        return links
    for a in getattr(cell, "find_all", lambda *_, **__: [])("a", href=True):
        text = _normspace(getattr(a, "get_text", lambda *_, **__: "")(" ", strip=True))
        href = a.get("href")
        if isinstance(href, str):
            links.append({"text": text, "href": href})
    return links


def _extract_infobox(soup: BeautifulSoup) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "kind": None,
        "type": None,
        "image": None,
        "kv_raw": {},
        "kv_links": {},
    }

    box = soup.select_one(".portable-infobox, section.portable-infobox")
    if box is not None:
        info["kind"] = "portable"
        title_el = box.select_one(".pi-title, h2.pi-item")
        info["type"] = _text_of(title_el) if title_el is not None else None
        img = box.select_one(".pi-image img, figure img")
        src = img.get("src") if img is not None else None
        info["image"] = src if isinstance(src, str) else None

        for row in box.select(".pi-data"):
            label = row.select_one(".pi-data-label")
            value = row.select_one(".pi-data-value")
            k = _text_of(label) if label is not None else None
            if not k:
                continue
            v = _value_text(value)
            if v:
                prev = info["kv_raw"].get(k)
                info["kv_raw"][k] = (
                    f"{prev}, {v}" if isinstance(prev, str) and prev else v
                )
            links = _value_links(value)
            if links:
                info["kv_links"].setdefault(k, []).extend(links)
        return info

    table = soup.select_one("table.infobox")
    if table is not None:
        info["kind"] = "table"
        cap = table.find("caption")
        info["type"] = (
            _text_of(cap)
            if cap is not None
            else (_text_of(table.find("th")) if table.find("th") else None)
        )
        img = table.select_one("img")
        src = img.get("src") if img is not None else None
        info["image"] = src if isinstance(src, str) else None

        for tr in table.find_all("tr"):
            th, td = tr.find("th"), tr.find("td")
            if th is None or td is None:
                continue
            k = _text_of(th)
            v = _value_text(td)
            if k and v:
                prev = info["kv_raw"].get(k)
                info["kv_raw"][k] = (
                    f"{prev}, {v}" if isinstance(prev, str) and prev else v
                )
                links = _value_links(td)
                if links:
                    info["kv_links"].setdefault(k, []).extend(links)
    return info


_SKIP_PARENTS = (
    "portable-infobox",
    "infobox",
    "toc",
    "navbox",
    "vertical-navbox",
    "metadata",
    "ambox",
    "mboverlay",
    "hatnote",
    "dablink",
    "sistersitebox",
)


def _has_forbidden_ancestor(tag: Tag) -> bool:
    p = tag.parent
    while isinstance(p, Tag):
        cls = p.get("class") or []
        if any(c in _SKIP_PARENTS for c in cls):
            return True
        p = p.parent
    return False


def _extract_lists(node: Tag) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for ul in node.find_all(["ul", "ol"], recursive=False):
        items = []
        for li in ul.find_all("li", recursive=False):
            nested = _extract_lists(li)
            # strip nested lists from this level's text capture
            for n in li.find_all(["ul", "ol"]):
                n.extract()
            items.append({"text": _text_of(li), "children": nested})
        blocks.append({"ordered": (ul.name == "ol"), "items": items})
    return blocks


def _is_section_heading(node: PageElement) -> bool:
    name = getattr(node, "name", None)
    return name in ("h2", "h3")


def _heading_text(node: PageElement) -> str:
    # We won't mutate; just read text. If edit spans leak in, they’ll be rare.
    return _text_of(node)


def _extract_table(tb: Tag) -> Dict[str, Any]:
    headers: List[str] = []
    header_row = tb.find("tr")
    if header_row and header_row.find_all(["th"]):  # type: ignore
        headers = [
            _normspace(c.get_text(" ", strip=True)) for c in header_row.find_all(["th"])  # type: ignore[misc]
        ]
    rows: List[List[str]] = []
    for tr in tb.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        rows.append([_normspace(c.get_text(" ", strip=True)) for c in cells])
    return {"headers": headers, "rows": rows}


def _extract_sections(content: Optional[PageElement]) -> List[Dict[str, Any]]:
    """
    Return only real sections (h2/h3 and below). Preface/lead paragraphs are *not*
    emitted as a 'Lead' section; they're available separately in `lead`.
    """
    sections: List[Dict[str, Any]] = []
    if content is None:
        return sections

    current: Optional[Dict[str, Any]] = None

    for child in getattr(content, "children", []):
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, (Tag, PageElement)):
            continue

        # Start a new section when we see the first h2/h3
        if _is_section_heading(child):
            if current is not None and any(
                current[k]
                for k in ("paragraphs", "lists", "tables", "figures", "quotes")
            ):
                sections.append(current)
            level = 2 if getattr(child, "name", None) == "h2" else 3
            current = {
                "level": level,
                "heading": _heading_text(child),
                "anchor": getattr(child, "get", lambda *_, **__: None)("id"),
                "paragraphs": [],
                "lists": [],
                "tables": [],
                "figures": [],
                "quotes": [],
            }
            continue

        # Ignore any content that appears before the first real heading
        if current is None:
            continue

        name = getattr(child, "name", None)
        if name == "p":
            txt = _text_of(child)
            if txt:
                current["paragraphs"].append(txt)
        elif name in ("ul", "ol"):
            current["lists"].append(
                {
                    "ordered": (name == "ol"),
                    "items": [
                        {"text": _text_of(li), "children": _extract_lists(li)}
                        for li in getattr(child, "find_all", lambda *_, **__: [])("li", recursive=False)
                    ],
                }
            )
        elif name == "table":
            current["tables"].append(_extract_table(child))  # type: ignore[arg-type]
        elif name == "blockquote":
            current["quotes"].append(_text_of(child))
        elif name == "figure":
            img = getattr(child, "find", lambda *_, **__: None)("img")
            src = getattr(img, "get", lambda *_, **__: None)("src") if img is not None else None
            cap = getattr(child, "find", lambda *_, **__: None)("figcaption")
            if isinstance(src, str):
                current["figures"].append({"src": src, "caption": _text_of(cap)})
        else:
            pass

    if current is not None and any(
        current[k] for k in ("paragraphs", "lists", "tables", "figures", "quotes")
    ):
        sections.append(current)
    return sections


def _collect_images_all(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if isinstance(src, str):
            out.append(src)
    # de-dupe preserving order
    seen: set[str] = set()
    res: List[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            res.append(u)
    return res


def _collect_internal_links(content: Optional[PageElement]) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    if content is None:
        return links
    for a in getattr(content, "find_all", lambda *_, **__: [])("a", href=True):
        href = a.get("href")
        if not isinstance(href, str):
            continue
        # internal if /wiki/... or relative (non-http, non-anchor-only)
        is_internal = href.startswith("/wiki/") or (
            not href.startswith("http") and not href.startswith("#")
        )
        if not is_internal:
            continue
        txt = _normspace(getattr(a, "get_text", lambda *_, **__: "")(" ", strip=True))
        links.append({"text": txt, "href": href})
    # de-dupe by pair
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for l in links:
        key = (l["text"], l["href"])
        if key not in seen:
            seen.add(key)
            out.append(l)
    return out


def _extract_meta(soup: BeautifulSoup) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "og": {
            "title": _meta_content(soup, "og:title", by="property"),
            "description": _meta_content(soup, "og:description", by="property"),
            "image": _meta_content(soup, "og:image", by="property"),
        },
        "twitter": {
            "title": _meta_content(soup, "twitter:title", by="name"),
            "description": _meta_content(soup, "twitter:description", by="name"),
            "image": _meta_content(soup, "twitter:image", by="name"),
        },
        "published": _meta_content(soup, "article:published_time", by="property"),
        "last_modified": _meta_content(soup, "article:modified_time", by="property")
        or _meta_content(soup, "og:updated_time", by="property"),
    }
    # JSON-LD blocks
    ld_all: List[Dict[str, Any]] = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            raw = s.string
            if not isinstance(raw, str):
                continue
            data = json.loads(raw)
            if isinstance(data, dict):
                ld_all.append(data)
        except Exception:
            continue
    if ld_all:
        meta["ld_json"] = ld_all
    return meta


def _infer_flags(categories: List[str]) -> Dict[str, bool]:
    cats_lower = [c.lower() for c in categories]
    return {
        "is_disambiguation": any("disambiguation" in c for c in cats_lower),
        "is_stub": any(c.endswith("stubs") or " stub" in c for c in cats_lower),
    }


def parse_article_html(html: str, fetched_at: Optional[str] = None) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    content = _content_root(soup)
    preface = _extract_preface(content)

    title = _first_heading(soup)
    if not title and soup.title and soup.title.string:
        title = _normspace(str(soup.title.string))

    canonical = _canonical_url(soup)
    lead = preface.get("paragraphs", [""])[0] if preface.get("paragraphs") else ""
    infobox = _extract_infobox(soup)
    sections = _extract_sections(content)
    categories = _extract_categories(soup)
    images_all = _collect_images_all(soup)
    links_internal = _collect_internal_links(content)
    meta = _extract_meta(soup)
    if fetched_at:
        meta["fetched_at"] = fetched_at

    raw_html_hash = _sha256_hex(html.encode("utf-8"))
    flags = _infer_flags(categories)

    entity_id = canonical if isinstance(canonical, str) else f"sha256:{raw_html_hash}"

    return {
        "id": entity_id,
        "title": title,
        "canonical_url": canonical,
        "raw_html_hash": raw_html_hash,
        "lead": lead,
        "preface": preface,  
        "infobox": infobox,  # {kind, type, image, kv_raw, kv_links}
        "sections": sections,
        "categories": categories,
        "images_all": images_all,
        "links_internal": links_internal,
        "meta": meta,
        "flags": flags,
        "provenance": {
            "parser_version": PARSER_VERSION,
            "notes": "MediaWiki/Fandom selectors; citation superscripts stripped.",
        },
    }


if __name__ == "__main__":
    import sys

    doc = parse_article_html(sys.stdin.read())
    print(json.dumps(doc, ensure_ascii=False, indent=2))
