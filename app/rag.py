# app/rag.py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

@dataclass
class DocChunk:
    doc_id: str
    title: str
    chunk_id: int
    text: str
    source_path: str

class SimpleRAG:
    def __init__(self, kb_dir: str = 'kb'):
        self.kb_dir = Path(kb_dir)
        self.chunks: List[DocChunk] = []
        self.tokens: List[List[str]] = []
        self.bm25 = None
        self._build_index()

    def _read_pdf(self, path: Path) -> str:
        try:
            r = PdfReader(str(path))
            return "\n".join((p.extract_text() or "") for p in r.pages)
        except Exception:
            return ""

    def _clean(self, t: str) -> str:
        return re.sub(r"\s+", " ", t).strip()

    def _split(self, text: str) -> List[str]:
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i+CHUNK_SIZE])
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _tok(self, s: str) -> List[str]:
        # Basit TR+EN tokenizasyon
        return re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", s.lower())

    def _build_index(self):
        pdfs = sorted(self.kb_dir.glob("*.pdf"))
        for p in pdfs:
            raw = self._clean(self._read_pdf(p))
            if not raw:
                continue
            parts = self._split(raw)
            for ci, part in enumerate(parts):
                self.chunks.append(DocChunk(
                    doc_id=p.name, title=p.stem, chunk_id=ci, text=part, source_path=str(p.name)
                ))
                self.tokens.append(self._tok(part))
        if self.tokens:
            self.bm25 = BM25Okapi(self.tokens)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[DocChunk, float]]:
        if not self.bm25:
            return []
        qtok = self._tok(query)
        scores = self.bm25.get_scores(qtok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idxs if scores[i] > 0.0]

    def topk_text(self, query: str, k: int = 5) -> str:
        ctx = self.retrieve(query, k=k)
        if not ctx:
            return ""
        joined = "\n".join([f"[{i+1}] {c.title}: {c.text}" for i,(c,_) in enumerate(ctx)])
        return joined[:2500]
