"""Download, verify, chunk, embed and store the policy wordings for RAG.

The PDFs are public policy forms that the Maine Bureau of Insurance publishes
"for general reference". They go in data/policies/ (git-ignored) and are never
redistributed. wordings.json pins each file's SHA-256: the wordings become text
the agent reads, so a swapped or tampered file is a prompt-injection route, and
ingest refuses anything that doesn't match.

Re-running replaces a wording's chunks in one transaction (idempotent).

Usage:
    python ingest.py --pdf-dir ../../data/policies --download
    (connects to DATABASE_URL, read from the repo's .env if not already set)
"""

import argparse
import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path

from chunking import chunk_pages, embedding_text

MANIFEST = Path(__file__).resolve().parent / "wordings.json"
# maine.gov rejects Python's default user agent.
USER_AGENT = "Mozilla/5.0 (compatible; ClaimLens-ingest/0.1)"


class IntegrityError(Exception):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pdf_path(pdf_dir, entry):
    return Path(pdf_dir) / f"{entry['wording_id']}.pdf"


def download(entry, pdf_dir):
    path = pdf_path(pdf_dir, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        entry["source_url"], headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise IntegrityError(
            f"{entry['wording_id']}: downloaded file doesn't match the pinned SHA-256"
        )
    path.write_bytes(data)
    return path


def verify(entry, path):
    if sha256(path) != entry["sha256"]:
        raise IntegrityError(
            f"{path}: SHA-256 doesn't match wordings.json; refusing to ingest"
        )


def extract_pages(path):
    from pypdf import PdfReader

    # pypdf warns about a font encoding in one wording; the text is still fine.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def store(conn, entry, n_pages, chunks, vectors):
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO policy_wordings (wording_id, title, insurer, form_number, source_url, sha256, pages)
            VALUES (%(wording_id)s, %(title)s, %(insurer)s, %(form_number)s, %(source_url)s, %(sha256)s, %(pages)s)
            ON CONFLICT (wording_id) DO UPDATE SET
                title = EXCLUDED.title, insurer = EXCLUDED.insurer, form_number = EXCLUDED.form_number,
                source_url = EXCLUDED.source_url, sha256 = EXCLUDED.sha256, pages = EXCLUDED.pages
            """,
            {**entry, "pages": n_pages},
        )
        conn.execute(
            "DELETE FROM policy_chunks WHERE wording_id = %s", (entry["wording_id"],)
        )
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO policy_chunks (wording_id, chunk_index, section, page_start, page_end, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        entry["wording_id"],
                        c["chunk_index"],
                        c["section"],
                        c["page_start"],
                        c["page_end"],
                        c["text"],
                        v,
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )


def ingest(conn, entry, pages, embedder):
    chunks = chunk_pages(pages)
    vectors = embedder.passages([embedding_text(c) for c in chunks])
    store(conn, entry, len(pages), chunks, vectors)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf-dir", required=True, help="Folder for the wording PDFs")
    parser.add_argument(
        "--download", action="store_true", help="Download missing PDFs first"
    )
    args = parser.parse_args()

    import psycopg
    from pgvector.psycopg import register_vector

    from embeddings import Embedder

    if "DATABASE_URL" not in os.environ:  # running on the host, not in a container
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv())  # searches upward from this file to the repo's .env
    entries = json.loads(MANIFEST.read_text())
    embedder = Embedder()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        for entry in entries:
            path = pdf_path(args.pdf_dir, entry)
            if not path.exists():
                if not args.download:
                    raise SystemExit(f"{path} missing; run with --download")
                download(entry, args.pdf_dir)
            verify(entry, path)
            n = ingest(conn, entry, extract_pages(path), embedder)
            print(f"{entry['wording_id']:>20}: {n} chunks")


if __name__ == "__main__":
    main()
