"""
Gallery Guide — Ingest Pipeline
Reads Art Institute of Chicago JSON files + Wikipedia enrichment,
builds rich text for each artwork, embeds with dense + sparse vectors,
and indexes into Qdrant.

Usage:
    python ingest.py --data-dir ../artic-api-data/json/artworks --limit 500
    python ingest.py --core-only   # only index the 8 famous core works
    python ingest.py --reset       # drop collection and re-index everything
"""

import argparse
import glob
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
COLLECTION   = "gallery_guide"
DENSE_MODEL  = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
QDRANT_HOST  = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", 6333))

HEADERS = {"User-Agent": "GalleryGuideBot/1.0 (educational project)"}

# ── Core famous works (always indexed) ────────────────────────────────
CORE_WORKS = [
    {
        "id": "mona_lisa",
        "type": "artwork",
        "title": "Mona Lisa",
        "artist": "Leonardo da Vinci",
        "year": "1503–1519",
        "culture": "Italian Renaissance",
        "medium": "Oil on poplar panel",
        "location": "Louvre Museum, Paris",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "text": """The Mona Lisa is a half-length portrait painting by Italian Renaissance artist Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance and has been described as the best known, the most visited, the most written about, the most sung about, and the most parodied work of art in the world.
The painting depicts a seated woman believed to be Lisa Gherardini, the wife of Florentine merchant Francesco del Giocondo. It is painted in oil on a white Lombardy poplar panel. Leonardo never gave the painting to the Giocondo family and kept it until his death.
The subject's mysterious smile has inspired countless analyses. Leonardo's use of sfumato — the subtle gradation of tone — is exemplified in the Mona Lisa, particularly in the face and hands. The painting was stolen from the Louvre in 1911 by an Italian worker named Vincenzo Peruggia, causing an international sensation. It was recovered two years later.""",
    },
    {
        "id": "sistine_chapel",
        "type": "artwork",
        "title": "Sistine Chapel Ceiling",
        "artist": "Michelangelo",
        "year": "1508–1512",
        "culture": "Italian Renaissance",
        "medium": "Fresco",
        "location": "Vatican Museums, Rome",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg/1280px-Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg",
        "text": """The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512, commissioned by Pope Julius II. The ceiling contains over 300 figures and is considered one of the greatest artistic achievements in Western civilization.
Michelangelo was primarily a sculptor and initially refused the commission. He worked lying on his back on scaffolding for four years. The most famous section is The Creation of Adam, showing God and Adam nearly touching fingers. The Pope threatened Michelangelo repeatedly when he felt the work was progressing too slowly.
The ceiling was restored between 1980 and 1994, revealing much brighter colors than previously thought, suggesting that centuries of candle soot had darkened the original vibrant palette.""",
    },
    {
        "id": "birth_of_venus",
        "type": "artwork",
        "title": "The Birth of Venus",
        "artist": "Sandro Botticelli",
        "year": "1484–1486",
        "culture": "Italian Renaissance",
        "medium": "Tempera on canvas",
        "location": "Uffizi Gallery, Florence",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/2/26/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg",
        "text": """The Birth of Venus by Sandro Botticelli depicts the goddess Venus arriving at the shore after her birth, standing on a giant scallop shell. It was commissioned by the Medici family in the mid-1480s.
The painting shows influences of classical antiquity and the idealized beauty typical of the Renaissance. Botticelli's Venus has an unusually long neck, sloping shoulders, and a tilted head that give her a graceful, otherworldly quality. The work represents a Neo-Platonic concept of divine love and beauty.
It is one of the most recognized images in Western art and represents a high point of Florentine Renaissance painting. Venus and Mars with Cupid and the Three Graces is a related work by Botticelli showing similar mythological themes.""",
    },
    {
        "id": "school_of_athens",
        "type": "artwork",
        "title": "The School of Athens",
        "artist": "Raphael",
        "year": "1509–1511",
        "culture": "Italian Renaissance",
        "medium": "Fresco",
        "location": "Vatican Museums, Rome",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg/1280px-%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg",
        "text": """The School of Athens is a fresco painted by Raphael between 1509 and 1511 as part of his commission to decorate the rooms of the Apostolic Palace in the Vatican. It depicts the great philosophers and scientists of ancient Greece gathered together.
At the center are Plato and Aristotle. Plato points upward, representing his theory of Forms, while Aristotle gestures downward, representing his empirical approach. Leonardo da Vinci was the model for Plato. Michelangelo appears as Heraclitus, seated alone in the foreground — Raphael added him after visiting the Sistine Chapel.
Pope Julius II originally wanted Raphael for the Sistine Chapel ceiling, but gave it to Michelangelo instead. Raphael immortalized his rival by including him in this masterpiece.""",
    },
    {
        "id": "last_supper",
        "type": "artwork",
        "title": "The Last Supper",
        "artist": "Leonardo da Vinci",
        "year": "1495–1498",
        "culture": "Italian Renaissance",
        "medium": "Tempera on gesso, pitch and mastic",
        "location": "Santa Maria delle Grazie, Milan",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/%C3%9Altima_Cena_-_Da_Vinci_5.jpg/1280px-%C3%9Altima_Cena_-_Da_Vinci_5.jpg",
        "text": """The Last Supper is a late 15th-century mural by Leonardo da Vinci depicting the moment Jesus announces that one of his apostles will betray him. It covers an end wall of the dining hall at the Convent of Santa Maria delle Grazie in Milan.
Leonardo used an experimental technique instead of traditional fresco, applying tempera directly to dry plaster. This caused the painting to deteriorate rapidly — what survives today is largely a restoration. The vanishing point of the perspective converges directly behind Jesus's head.
The painting is 460 cm × 880 cm and shows the psychological reactions of each apostle. Leonardo is said to have left the face of Judas incomplete for years, struggling to capture the essence of a traitor.""",
    },
    {
        "id": "david_michelangelo",
        "type": "artwork",
        "title": "David",
        "artist": "Michelangelo",
        "year": "1501–1504",
        "culture": "Italian Renaissance",
        "medium": "Marble sculpture",
        "location": "Galleria dell'Accademia, Florence",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Michelangelo%27s_David_-_right_view_2.jpg/800px-Michelangelo%27s_David_-_right_view_2.jpg",
        "text": """David is a 5.17-metre marble statue created by Michelangelo between 1501 and 1504. Unlike earlier depictions showing David victorious after battle, Michelangelo chose to show him before the fight with Goliath, tense and contemplating his forthcoming challenge.
The statue weighs approximately 5,660 kg. Michelangelo was just 26 years old when he began this work. The right hand is slightly larger than the left, interpreted as a symbol of divine strength. The veins in the right hand are rendered with extraordinary anatomical detail.
Originally placed in the Piazza della Signoria, it has been at the Galleria dell'Accademia since 1873. The original block of marble had been abandoned for 25 years before Michelangelo was commissioned to work with it.""",
    },
    {
        "id": "arnolfini_portrait",
        "type": "artwork",
        "title": "The Arnolfini Portrait",
        "artist": "Jan van Eyck",
        "year": "1434",
        "culture": "Northern Renaissance",
        "medium": "Oil on oak panel",
        "location": "National Gallery, London",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Van_Eyck_-_Arnolfini_Portrait.jpg/800px-Van_Eyck_-_Arnolfini_Portrait.jpg",
        "text": """The Arnolfini Portrait by Jan van Eyck shows a wealthy Italian merchant Giovanni di Nicolao di Arnolfini and his wife in their home in Bruges. The painting is signed above the mirror: Johannes de Eyck fuit hic — Jan van Eyck was here.
The convex mirror on the back wall reflects two additional figures, one of whom may be van Eyck himself. The single lit candle may symbolize the presence of God or the marriage ceremony. Van Eyck's mastery of oil paint allowed him to render textures, light, and reflections with unprecedented realism.
The painting has been debated for centuries — whether it depicts a marriage ceremony, a betrothal, or simply a portrait. The woman's apparent pregnancy may simply reflect 15th-century fashion rather than actual pregnancy.""",
    },
    {
        "id": "venus_of_urbino",
        "type": "artwork",
        "title": "Venus of Urbino",
        "artist": "Titian",
        "year": "1538",
        "culture": "Italian Renaissance",
        "medium": "Oil on canvas",
        "location": "Uffizi Gallery, Florence",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Tiziano_-_Venere_di_Urbino_-_Google_Art_Project.jpg/1280px-Tiziano_-_Venere_di_Urbino_-_Google_Art_Project.jpg",
        "text": """The Venus of Urbino by Titian shows a nude woman reclining on white sheets and looking directly at the viewer with a confident, relaxed expression. It was commissioned by Guidobaldo della Rovere, Duke of Urbino, in 1538.
Titian's handling of paint is extraordinary — the warm flesh tones contrasted against cool white sheets demonstrate his mastery of color. Mark Twain called it the most licentious painting in existence. The small dog at the woman's feet symbolizes loyalty or fidelity.
The painting inspired many subsequent reclining female nudes in Western art history, including Manet's Olympia (1865), which deliberately echoed its composition to provocative effect. The work is considered a pinnacle of Venetian Renaissance painting.""",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def get_wikipedia_text(title: str, artist: str) -> str:
    """Fetch Wikipedia extract for an artwork."""
    for query in [title, f"{title} {artist}", f"{title} painting"]:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
                headers=HEADERS, timeout=10,
            )
            results = r.json().get("query", {}).get("search", [])
            if not results:
                continue
            page_title = results[0]["title"]
            r2 = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "titles": page_title, "prop": "extracts", "explaintext": True, "format": "json"},
                headers=HEADERS, timeout=10,
            )
            pages = r2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if len(extract) > 300:
                    return extract[:3000]
        except Exception:
            pass
        time.sleep(0.3)
    return ""


def build_text(data: dict) -> str:
    """Build rich text from Art Institute JSON record."""
    parts = []
    title   = data.get("title", "")
    artist  = data.get("artist_title", "Unknown")
    date    = data.get("date_display", "")
    medium  = data.get("medium_display", "")
    origin  = data.get("place_of_origin", "")
    dept    = data.get("department_title", "")
    desc    = strip_html(data.get("description", ""))
    prov    = (data.get("provenance_text") or "")[:400]
    subjects = ", ".join(data.get("subject_titles") or [])
    techs    = ", ".join(data.get("technique_titles") or [])
    gallery  = data.get("gallery_title") or ""
    on_view  = data.get("is_on_view", False)

    if title:  parts.append(f"{title} is a Renaissance artwork by {artist}.")
    if date:   parts.append(f"Created {date}.")
    if origin: parts.append(f"Origin: {origin}.")
    if medium: parts.append(f"Medium: {medium}.")
    if techs:  parts.append(f"Technique: {techs}.")
    if subjects: parts.append(f"Subjects: {subjects}.")
    if desc:   parts.append(f"Description: {desc[:600]}")
    if prov:   parts.append(f"Provenance: {prov}")
    if gallery and on_view: parts.append(f"Currently on view in {gallery} at the Art Institute of Chicago.")

    return " ".join(parts)


def is_renaissance(data: dict) -> bool:
    style   = data.get("style_title", "") or ""
    styles  = data.get("style_titles") or []
    dept    = data.get("department_title", "") or ""
    ds      = data.get("date_start") or 0
    pub_dom = data.get("is_public_domain", False)
    img     = data.get("image_id", "")

    if not pub_dom or not img:
        return False

    renaiss = "Renaissance" in style or any("Renaissance" in s for s in styles)
    period  = (dept == "Painting and Sculpture of Europe" and 1300 <= ds <= 1620)
    return renaiss or period


# ── Embedder ───────────────────────────────────────────────────────────

class Embedder:
    def __init__(self):
        print("Loading embedding models…")
        self.dense  = TextEmbedding(DENSE_MODEL)
        self.sparse = SparseTextEmbedding(SPARSE_MODEL)

    def embed(self, text: str) -> tuple[list[float], SparseVector]:
        d = list(self.dense.embed([text]))[0].tolist()
        s = list(self.sparse.embed([text]))[0]
        return d, SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


# ── Qdrant setup ───────────────────────────────────────────────────────

def get_client():
    return QdrantClient(
        url=f"https://{QDRANT_HOST}",
        api_key=os.getenv("QDRANT_API_KEY", ""),
    )


def setup_collection(client: QdrantClient, reset: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        if reset:
            client.delete_collection(COLLECTION)
            print(f"Dropped collection '{COLLECTION}'")
        else:
            print(f"Collection '{COLLECTION}' already exists — use --reset to re-index")
            return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
    )
    print(f"Created collection '{COLLECTION}'")


def upsert_batch(client: QdrantClient, points: list[PointStruct]):
    client.upsert(collection_name=COLLECTION, points=points)


# ── Ingest helpers ────────────────────────────────────────────────────

def make_point(embedder: Embedder, doc: dict) -> PointStruct | None:
    text = doc.get("text", "")
    if not text.strip():
        return None
    try:
        dense, sparse = embedder.embed(text)
        return PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": dense, "sparse": sparse},
            payload={
                "doc_id":      doc["id"],
                "type":        doc.get("type", "artwork"),
                "title":       doc.get("title", ""),
                "artist":      doc.get("artist", ""),
                "year":        doc.get("year", ""),
                "medium":      doc.get("medium", ""),
                "culture":     doc.get("culture", ""),
                "location":    doc.get("location", ""),
                "image_url":   doc.get("image_url", ""),
                "text":        text[:2000],
                "subjects":    doc.get("subjects", ""),
                "gallery":     doc.get("gallery", ""),
                "is_on_view":  doc.get("is_on_view", False),
            },
        )
    except Exception as e:
        print(f"  Error embedding {doc.get('id')}: {e}")
        return None


# ── Main ingest functions ─────────────────────────────────────────────

def ingest_core(client: QdrantClient, embedder: Embedder) -> int:
    print(f"\nIndexing {len(CORE_WORKS)} core works…")
    points = []
    for work in CORE_WORKS:
        # Enrich with Wikipedia
        wiki = get_wikipedia_text(work["title"], work["artist"])
        if wiki:
            work["text"] = work["text"] + "\n\n" + wiki[:1500]
        p = make_point(embedder, work)
        if p:
            points.append(p)
            print(f"  ✓ {work['title']}")
        time.sleep(0.2)

    if points:
        upsert_batch(client, points)
    return len(points)


def ingest_artic(client: QdrantClient, embedder: Embedder, data_dir: str, limit: int = 400, enrich_wiki: bool = True) -> int:
    files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"\nScanning {len(files)} Art Institute files…")

    # Get existing doc_ids to avoid duplicates
    existing = set()
    try:
        scroll = client.scroll(collection_name=COLLECTION, limit=10000, with_payload=["doc_id"])
        for point in scroll[0]:
            existing.add(point.payload.get("doc_id", ""))
    except Exception:
        pass

    points = []
    indexed = skipped = errors = 0

    for filepath in files:
        if indexed >= limit:
            break
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not is_renaissance(data):
                continue

            doc_id = f"artic_{data['id']}"
            if doc_id in existing:
                skipped += 1
                continue

            image_id  = data.get("image_id", "")
            image_url = f"https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg" if image_id else ""

            text = build_text(data)

            # Wikipedia enrichment (optional, slower)
            if enrich_wiki and data.get("title") and data.get("artist_title"):
                wiki = get_wikipedia_text(data["title"], data["artist_title"])
                if wiki:
                    text += "\n\n" + wiki[:1000]
                time.sleep(0.3)

            doc = {
                "id":       doc_id,
                "type":     "artwork",
                "title":    data.get("title", ""),
                "artist":   data.get("artist_title", "Unknown"),
                "year":     data.get("date_display", ""),
                "medium":   data.get("medium_display", ""),
                "culture":  data.get("place_of_origin", ""),
                "location": data.get("department_title", ""),
                "image_url": image_url,
                "text":     text,
                "subjects": ", ".join(data.get("subject_titles") or []),
                "gallery":  data.get("gallery_title") or "",
                "is_on_view": data.get("is_on_view", False),
            }

            p = make_point(embedder, doc)
            if p:
                points.append(p)
                indexed += 1
                print(f"  ✓ [{indexed}/{limit}] {doc['title'][:50]}")

            # Batch upsert every 50
            if len(points) >= 50:
                upsert_batch(client, points)
                points = []

        except Exception as e:
            errors += 1
            print(f"  ✗ Error: {e}")

    if points:
        upsert_batch(client, points)

    print(f"\nArt Institute: {indexed} indexed, {skipped} skipped, {errors} errors")
    return indexed


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gallery Guide ingest pipeline")
    parser.add_argument("--data-dir",   default="../artic-api-data/json/artworks", help="Path to Art Institute JSON files")
    parser.add_argument("--limit",      type=int, default=400, help="Max artworks to index from Art Institute")
    parser.add_argument("--reset",      action="store_true", help="Drop and recreate collection")
    parser.add_argument("--core-only",  action="store_true", help="Only index core famous works")
    parser.add_argument("--no-wiki",    action="store_true", help="Skip Wikipedia enrichment (faster)")
    args = parser.parse_args()

    client   = get_client()
    embedder = Embedder()

    setup_collection(client, reset=args.reset)

    total = 0
    total += ingest_core(client, embedder)

    if not args.core_only:
        if os.path.exists(args.data_dir):
            total += ingest_artic(client, embedder, args.data_dir, limit=args.limit, enrich_wiki=not args.no_wiki)
        else:
            print(f"Warning: data directory '{args.data_dir}' not found — only core works indexed")

    print(f"\n✓ Done! {total} documents indexed into '{COLLECTION}'")


if __name__ == "__main__":
    main()
