"""Collect openly licensed arXiv papers (metadata via OAI-PMH, PDFs from arxiv.org).

    python -m grasp_train.collect_arxiv --per-field 100

OAI-PMH is used instead of the search API because it reports each paper's licence. Requests
are spaced 3 seconds apart, as arXiv asks of automated clients. Re-running resumes: papers
already in metadata.jsonl or already downloaded are skipped.
"""
from __future__ import annotations

import argparse
import random
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import requests

from .common import DATA, append_jsonl, read_jsonl

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
USER_AGENT = "GRASP-SLM research data collection (university project; contact via GitHub)"

# OAI set -> primary-category prefixes that count as that field.
FIELDS = {
    "cs": ("cs.",),
    "q-bio": ("q-bio.",),
    "physics": ("physics.", "cond-mat.", "astro-ph.", "quant-ph", "hep-", "gr-qc", "nucl-", "nlin."),
    "eess": ("eess.",),
}
OPEN_LICENSES = ("creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
                 "creativecommons.org/publicdomain/zero/")
NC_LICENSES = ("creativecommons.org/licenses/by-nc/", "creativecommons.org/licenses/by-nc-sa/")


class Client:
    def __init__(self, delay: float = 3.0):
        self.delay = delay
        self.last = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, url: str, **params) -> requests.Response:
        for attempt in range(6):
            wait = self.last + self.delay - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.session.get(url, params=params or None, timeout=120)
            except requests.RequestException as e:
                print(f"  network error ({e}); retrying")
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code in (429, 503):
                retry = int(r.headers.get("Retry-After", "30"))
                print(f"  server busy ({r.status_code}); waiting {retry}s")
                time.sleep(retry)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"Giving up on {url}")


def list_records(client: Client, oai_set: str, start: date, end: date):
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": oai_set,
              "from": start.isoformat(), "until": end.isoformat()}
    while True:
        root = ET.fromstring(client.get(OAI, **params).content)
        for rec in root.iterfind(".//oai:record", NS):
            meta = rec.find("oai:metadata/ax:arXiv", NS)
            if meta is not None:
                yield {
                    "arxiv_id": meta.findtext("ax:id", "", NS).strip(),
                    "title": " ".join(meta.findtext("ax:title", "", NS).split()),
                    "abstract": " ".join(meta.findtext("ax:abstract", "", NS).split()),
                    "categories": meta.findtext("ax:categories", "", NS).split(),
                    "license": meta.findtext("ax:license", "", NS).strip(),
                    "created": meta.findtext("ax:created", "", NS).strip(),
                }
        token = root.find(".//oai:resumptionToken", NS)
        if token is None or not (token.text or "").strip():
            return
        params = {"verb": "ListRecords", "resumptionToken": token.text.strip()}


def is_open(license_url: str, allow_nc: bool) -> bool:
    allowed = OPEN_LICENSES + (NC_LICENSES if allow_nc else ())
    return any(a in license_url for a in allowed)


def paper_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", default=list(FIELDS), choices=list(FIELDS))
    ap.add_argument("--per-field", type=int, default=100, help="papers per field (proposal: 300-800 in total)")
    ap.add_argument("--min-year", type=int, default=2021)
    ap.add_argument("--window-days", type=int, default=2, help="size of each random OAI date window")
    ap.add_argument("--allow-nc", action="store_true", help="also accept CC BY-NC and BY-NC-SA papers")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    pdf_dir = out / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    have = {r["paper_id"]: r for r in read_jsonl(meta_path)}
    rng = random.Random(args.seed)
    client = Client()

    for field in args.fields:
        prefixes = FIELDS[field]
        count = sum(1 for r in have.values() if r["field"] == field)
        print(f"== {field}: {count}/{args.per_field} already collected")
        tried_windows = 0
        while count < args.per_field and tried_windows < 200:
            tried_windows += 1
            span = (date.today() - date(args.min_year, 1, 1)).days - 30
            start = date(args.min_year, 1, 1) + timedelta(days=rng.randrange(max(1, span)))
            end = start + timedelta(days=args.window_days - 1)
            candidates = [r for r in list_records(client, field, start, end)
                          if r["categories"] and r["categories"][0].startswith(prefixes)
                          and is_open(r["license"], args.allow_nc)
                          and int(r["created"][:4] or 0) >= args.min_year
                          and paper_id(r["arxiv_id"]) not in have]
            rng.shuffle(candidates)
            # A few papers per window keeps the sample spread over time.
            for rec in candidates[: max(1, args.per_field // 20)]:
                if count >= args.per_field:
                    break
                pid = paper_id(rec["arxiv_id"])
                pdf = pdf_dir / f"{pid}.pdf"
                try:
                    r = client.get(f"https://arxiv.org/pdf/{rec['arxiv_id']}")
                except Exception as e:  # noqa: BLE001 - keep collecting on single failures
                    print(f"  skip {pid}: {e}")
                    continue
                if not r.content.startswith(b"%PDF"):
                    print(f"  skip {pid}: not a PDF")
                    continue
                pdf.write_bytes(r.content)
                row = {"paper_id": pid, "field": field, **rec, "pdf": f"pdfs/{pid}.pdf"}
                append_jsonl(meta_path, row)
                have[pid] = row
                count += 1
                print(f"  [{field} {count}/{args.per_field}] {pid} {rec['title'][:70]}")
    print(f"Done: {len(have)} papers in {meta_path}")


if __name__ == "__main__":
    main()
