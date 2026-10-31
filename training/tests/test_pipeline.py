import json
import random

import pytest

from grasp_train import build_datasets
from grasp_train.common import ROOT, SCHEMA, canonical, chatml, schema_errors, to_target, unknown_citations, write_json
from grasp_train.metrics import citation_validity, numbers, numeric_accuracy, parse, schema_valid
from grasp_train.perturb import KINDS, make_rejected

GOOD = {
    "core_problem": {"citations": ["c1"], "text": "Adapting large models is costly."},
    "proposed_method": {"text": "Sparse adapters.", "citations": ["c2"]},
    "key_metrics": [{"metric": "F1", "value": "92.7", "dataset": "SQuAD v1.1", "citations": ["c3"]},
                    {"metric": "EM", "value": "87.4%", "dataset": "SQuAD v1.1", "citations": ["c3"]}],
    "limitations": [{"text": "English only.", "citations": ["c4"]}],
}


def test_schema_accepts_good_and_rejects_bad():
    assert schema_errors(GOOD) == []
    bad = json.loads(json.dumps(GOOD))
    bad["core_problem"]["citations"] = ["chunk1"]
    del bad["limitations"]
    assert len(schema_errors(bad)) == 2


def test_canonical_orders_keys_like_the_schema():
    c = canonical(GOOD)
    assert list(c) == list(SCHEMA["properties"])
    assert list(c["core_problem"]) == ["text", "citations"]
    assert list(c["key_metrics"][0]) == ["metric", "value", "dataset", "citations"]
    assert to_target(GOOD).startswith('{"core_problem": {"text": "Adapting')


def test_chatml_matches_java_prompt_format():
    java = (ROOT / "app/src/main/java/lk/ac/rjt/grasp/llm/ChatMl.java").read_text(encoding="utf-8")
    assert '"<|im_start|>system\\n"' in java and '"<|im_start|>assistant\\n"' in java
    assert chatml("S", "U") == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"


def test_unknown_citations():
    assert unknown_citations(GOOD, ["c1", "c2", "c3"]) == ["c4"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_perturbation_is_a_worse_answer(kind):
    rng = random.Random(0)
    chosen = to_target(GOOD)
    rejected = make_rejected(canonical(GOOD), kind, rng)
    assert rejected and rejected != chosen
    obj = parse(rejected)
    if kind in ("malformed_json", "missing_citation"):
        assert not schema_valid(obj)
    elif kind == "altered_number":
        assert schema_valid(obj) and obj != canonical(GOOD)
    else:
        assert len(obj["key_metrics"]) == len(GOOD["key_metrics"]) + 1


def test_numbers_are_normalised():
    # Numbers glued to letters (version labels such as "v1.1", model names such as "GPT4") are not values.
    assert numbers("acc 92.7% on 1,024 images, SQuAD v1.1") == {92.7, 1024.0}


def test_numeric_accuracy_detects_hallucinated_values():
    source = "We obtain an F1 of 92.7 and EM of 87.4 on SQuAD v1.1."
    assert numeric_accuracy(GOOD, GOOD, source)["numeric_f1"] == 1.0
    wrong = json.loads(json.dumps(GOOD))
    wrong["key_metrics"][0]["value"] = "93.7"
    m = numeric_accuracy(wrong, GOOD, source)
    assert m["grounded_precision"] == 0.5 and m["reference_recall"] == 0.5


def test_citation_validity():
    assert citation_validity(GOOD, ["c1", "c2", "c3", "c4"]) == 1.0
    assert citation_validity(GOOD, ["c1"]) == 0.2
    assert citation_validity(None, []) is None


def test_build_datasets_end_to_end(tmp_path, monkeypatch):
    export, drafts, verified, out = tmp_path / "export.jsonl", tmp_path / "drafts", tmp_path / "verified", tmp_path / "ds"
    rows = []
    for i in range(20):
        pid = f"2401.{i:05d}"
        rows.append({"paper_id": pid, "source_file": f"{pid}.pdf", "title": f"Paper {i}", "num_pages": 10,
                     "chunks": [{"id": f"c{k}", "text": f"chunk {k} F1 92.7 EM 87.4"} for k in range(6)],
                     "context_chunk_ids": ["c1", "c2", "c3", "c4"],
                     "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": f"U{i}"}]})
        write_json(drafts / f"{pid}.json", {"valid": True, "analysis": GOOD, "teacher": {"model": "t"}})
        if i < 6:
            write_json(verified / f"{pid}.json", {"status": "verified", "analysis": GOOD})
    write_json(verified / "2401.00019.json", {"status": "rejected", "analysis": None})
    export.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["x", "--export", str(export), "--drafts", str(drafts),
                                     "--verified", str(verified), "--out", str(out)])
    build_datasets.main()

    split = json.loads((out / "split.json").read_text())
    assert len(split["test"]) == 2 and all(int(p[-5:]) < 6 for p in split["test"])  # test = verified only
    assert "2401.00019" not in split["train"] + split["val"] + split["test"]  # rejected excluded
    assert not set(split["train"]) & set(split["test"])
    sft = [json.loads(l) for l in (out / "sft_train.jsonl").read_text().splitlines()]
    assert sft[0]["messages"][2]["content"] == to_target(GOOD)
    dpo = [json.loads(l) for l in (out / "dpo_train.jsonl").read_text().splitlines()]
    assert dpo and all(r["chosen"] != r["rejected"] for r in dpo)
