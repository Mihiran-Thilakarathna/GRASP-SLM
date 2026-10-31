"""Local web tool for manually checking and correcting teacher drafts (proposal section 6.3).

    python -m grasp_train.review_app        # then open http://127.0.0.1:7861

For each paper you see the context chunks the model was given and the draft JSON. Fix anything
that is not supported by the chunks, then press "Save as verified". Papers that cannot be
labelled (bad PDF parse, not a research paper) can be rejected. Results go to data/verified/.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from .common import DATA, canonical, read_json, read_jsonl, schema_errors, unknown_citations, write_json

GUIDE = """**How to verify a draft**
1. Read the chunks on the left. Every statement in the JSON must be supported by the chunks it cites.
2. Check every number in `key_metrics` character by character against the chunk text.
3. Remove metrics or limitations that are not in the chunks; add important ones that are missing.
4. Citations must be ids of the chunks shown (for example `"c3"`).
5. Press **Save as verified**. Use **Reject paper** for broken PDFs or non-research documents."""


class Store:
    def __init__(self, export: Path, drafts: Path, verified: Path):
        self.records = {r["paper_id"]: r for r in read_jsonl(export)}
        self.drafts, self.verified = drafts, verified
        verified.mkdir(parents=True, exist_ok=True)

    def ids(self) -> list[str]:
        return sorted(pid for pid in self.records if (self.drafts / f"{pid}.json").exists())

    def status(self, pid: str) -> str:
        path = self.verified / f"{pid}.json"
        return read_json(path)["status"] if path.exists() else "unreviewed"

    def counts(self) -> str:
        statuses = [self.status(p) for p in self.ids()]
        return (f"**{statuses.count('verified')} verified** (target: at least 100) · "
                f"{statuses.count('rejected')} rejected · {statuses.count('unreviewed')} unreviewed")

    def choices(self, show: str) -> list[str]:
        return [p for p in self.ids() if show == "all" or self.status(p) == show]

    def load(self, pid: str) -> tuple[str, str, str, str]:
        rec = self.records[pid]
        context = set(rec["context_chunk_ids"])
        chunks = [c for c in rec["chunks"] if c["id"] in context]
        evidence = "\n\n---\n\n".join(
            f"**[{c['id']}]** · {c['section']} · p. {c['page_start']}"
            + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "") + f"\n\n{c['text']}"
            for c in chunks)
        vpath = self.verified / f"{pid}.json"
        if vpath.exists() and read_json(vpath).get("analysis"):
            analysis, source = read_json(vpath)["analysis"], "your saved version"
        else:
            draft = read_json(self.drafts / f"{pid}.json")
            analysis, source = draft.get("analysis") or {}, f"teacher draft ({draft['teacher']['model']})"
        header = f"### {rec['title']}\n`{pid}` · {rec['num_pages']} pages · status: **{self.status(pid)}** · showing {source}"
        return header, evidence, json.dumps(analysis, ensure_ascii=False, indent=2), self.counts()

    def save(self, pid: str, text: str, notes: str) -> str:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            return f"Not saved: invalid JSON ({e})"
        errors = schema_errors(obj)
        unknown = unknown_citations(obj, self.records[pid]["context_chunk_ids"]) if not errors else []
        if errors or unknown:
            return "Not saved:\n- " + "\n- ".join(errors + [f"unknown citation {c}" for c in unknown])
        write_json(self.verified / f"{pid}.json", {
            "paper_id": pid, "status": "verified", "analysis": canonical(obj), "notes": notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return f"Saved {pid} as verified."

    def reject(self, pid: str, notes: str) -> str:
        write_json(self.verified / f"{pid}.json", {
            "paper_id": pid, "status": "rejected", "analysis": None, "notes": notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return f"Rejected {pid}."

    def next_unreviewed(self, after: str | None) -> str | None:
        ids = self.ids()
        start = ids.index(after) + 1 if after in ids else 0
        for pid in ids[start:] + ids[:start]:
            if self.status(pid) == "unreviewed":
                return pid
        return None


def build(store: Store) -> gr.Blocks:
    with gr.Blocks(title="GRASP-SLM review") as demo:
        gr.Markdown("## GRASP-SLM label review")
        counts = gr.Markdown(store.counts())
        with gr.Row():
            show = gr.Radio(["unreviewed", "verified", "rejected", "all"], value="unreviewed", label="Show")
            paper = gr.Dropdown(choices=store.choices("unreviewed"), label="Paper", scale=3)
            nxt = gr.Button("Next unreviewed")
        header = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Evidence (context chunks)")
                evidence = gr.Markdown(height=650)
            with gr.Column(scale=1):
                gr.Markdown(GUIDE)
                editor = gr.Code(language="json", label="Analysis JSON", lines=28)
                notes = gr.Textbox(label="Notes (optional)", lines=1)
                with gr.Row():
                    save = gr.Button("Save as verified", variant="primary")
                    reject = gr.Button("Reject paper", variant="stop")
                message = gr.Markdown()

        def on_select(pid):
            if not pid:
                return "", "", "", store.counts()
            return store.load(pid)

        def on_show(which):
            return gr.Dropdown(choices=store.choices(which), value=None)

        def on_next(pid):
            n = store.next_unreviewed(pid)
            return gr.Dropdown(choices=store.choices("unreviewed"), value=n), "unreviewed"

        def on_save(pid, text, note):
            msg = store.save(pid, text, note) if pid else "Choose a paper first."
            return msg, store.counts()

        def on_reject(pid, note):
            msg = store.reject(pid, note) if pid else "Choose a paper first."
            return msg, store.counts()

        paper.change(on_select, paper, [header, evidence, editor, counts])
        # .input fires only on user clicks, so "Next unreviewed" can switch the filter without clearing the paper.
        show.input(on_show, show, paper)
        nxt.click(on_next, paper, [paper, show])
        save.click(on_save, [paper, editor, notes], [message, counts])
        reject.click(on_reject, [paper, notes], [message, counts])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=str(DATA / "export.jsonl"))
    ap.add_argument("--drafts", default=str(DATA / "drafts"))
    ap.add_argument("--verified", default=str(DATA / "verified"))
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    store = Store(Path(args.export), Path(args.drafts), Path(args.verified))
    if not store.ids():
        raise SystemExit("No drafts found. Run grasp_train.draft_labels first.")
    build(store).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
