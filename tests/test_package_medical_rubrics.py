from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List


def _load_package_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "package_medical_rubrics.py"
    spec = importlib.util.spec_from_file_location("package_medical_rubrics", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_pkg = _load_package_module()


def _example_doc(
    *, source: str, source_id: str, pair_label: str, n_rubrics: int = 2
) -> Dict[str, Any]:
    example_id = f"{source}__{source_id}"
    rubrics: List[Dict[str, Any]] = []
    for i in range(n_rubrics):
        rubrics.append(
            {
                "rubric_id": f"{example_id}__rubric_{i}_{i}_0",
                "text": f"Rubric {i} for {source_id}",
                "source_stage": "initial_seed" if i == 0 else "initial",
                "depth": 0,
                "parent_id": None,
                "accepted": True,
                "metadata": {"foo": "bar"},
            }
        )
    candidates = [
        {
            "candidate_id": f"{example_id}__pair_a",
            "source_label": "pair_response_a",
            "origin_kind": "pair_anchor",
            "text": "A answer text",
        },
        {
            "candidate_id": f"{example_id}__pair_b",
            "source_label": "pair_response_b",
            "origin_kind": "pair_anchor",
            "text": "B answer text",
        },
        {
            "candidate_id": f"{example_id}__generated_0",
            "source_label": "generated_direct",
            "origin_kind": "generated",
            "text": "C alt answer text",
        },
    ]
    evaluations: List[Dict[str, Any]] = []
    for r in rubrics:
        for c in candidates:
            evaluations.append(
                {
                    "rubric_id": r["rubric_id"],
                    "candidate_id": c["candidate_id"],
                    "satisfied": (c["source_label"] == "pair_response_b"),
                    "reasoning": f"reason for {c['source_label']} on {r['rubric_id']}" * 30,
                }
            )
    method_block = {
        "ranking": [
            {"candidate_id": candidates[1]["candidate_id"], "rank": 1, "score": 0.9},
            {"candidate_id": candidates[0]["candidate_id"], "rank": 2, "score": 0.5},
            {"candidate_id": candidates[2]["candidate_id"], "rank": 3, "score": 0.1},
        ],
        "rubrics": rubrics,
        "evaluations": evaluations,
        "production_bank": [
            {
                "production_rubric_id": "production_0",
                "group_id": "treatment_decision",
                "label": "Treatment Decision",
                "family": "procedure_and_intervention",
                "canonical_text": "If a treatment was selected, the note records the decision.",
                "text": "If a treatment was selected, the note records the decision.",
                "conditionality": "if_discussed",
                "importance_tier": "major",
                "action_taken": "kept",
                "source_member_count": 1,
                "coverage_count": 1.0,
                "discrimination_score": 0.0,
            }
        ],
        "artifact": {
            "initial_rubric_count": n_rubrics,
            "initial_seed_rubric_count": 1,
            "seed_rubric_input_count": 1,
            "seed_rubric_accepted_count": 1,
            "seed_rubric_rejected_count": 0,
            "final_rubric_count": n_rubrics,
        },
    }
    return {
        "example": {
            "example_id": example_id,
            "source": source,
            "task_prompt": f"Question for {source_id}?",
            "pair_response_a": "A answer text",
            "pair_response_b": "B answer text",
            "pair_correct_label": pair_label,
        },
        "candidates": candidates,
        "methods": {
            "rrd_uniform": method_block,
            "rrd_whitened_uniform": method_block,
        },
    }


class PackageMedicalRubricsTest(unittest.TestCase):
    def test_rubric_and_evaluation_outputs_align_with_slim_file(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            (run_dir / "examples").mkdir(parents=True)

            sources = [
                ("medical_a", "0006001-aaaaaa", "a"),
                ("medical_b", "0006002-bbbbbb", "b"),
                ("medical_c", "0006003-cccccc", "b"),
            ]
            for source, sid, label in sources:
                doc = _example_doc(source=source, source_id=sid, pair_label=label)
                fp = run_dir / "examples" / f"{source}__{sid}.json"
                fp.write_text(json.dumps(doc), encoding="utf-8")

            slim_path = tmp_path / "slim.jsonl"
            with slim_path.open("w", encoding="utf-8") as f:
                for _, sid, _label in sources:
                    f.write(json.dumps({"id": sid, "question": "..."}) + "\n")

            out_dir = tmp_path / "out"
            rc = _pkg.main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--align-with",
                    str(slim_path),
                    "--out-dir",
                    str(out_dir),
                    "--stem",
                    "smoke",
                    "--reasoning-char-limit",
                    "50",
                ]
            )
            self.assertEqual(rc, 0)

            rubrics_path = out_dir / "smoke_rubrics.jsonl"
            eval_path = out_dir / "smoke_rubric_evaluations.jsonl"
            self.assertTrue(rubrics_path.exists())
            self.assertTrue(eval_path.exists())

            rubric_rows = [
                json.loads(line)
                for line in rubrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            eval_rows = [
                json.loads(line)
                for line in eval_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rubric_rows), 3)
            self.assertEqual([r["id"] for r in rubric_rows], [s[1] for s in sources])

            first = rubric_rows[0]
            self.assertEqual(first["rubric_count"], 2)
            self.assertEqual(first["gold_label"], "a")
            self.assertEqual(first["correct_answer"], "reference_answer_a")
            self.assertEqual(len(first["production_bank"]), 1)
            self.assertEqual(len(first["ranking"]), 3)
            stages = {r["source_stage"] for r in first["rubrics"]}
            self.assertEqual(stages, {"initial_seed", "initial"})
            self.assertNotIn("metadata", first["rubrics"][0])
            self.assertNotIn("accepted", first["rubrics"][0])

            self.assertEqual(len(eval_rows), 3 * 2 * 3)
            sample_ev = eval_rows[0]
            self.assertIn("rubric_text", sample_ev)
            self.assertIn("candidate_text", sample_ev)
            self.assertIn("candidate_role", sample_ev)
            self.assertLessEqual(len(sample_ev["reasoning"]), 50 + 3)

            roles_for_first = {
                ev["candidate_role"]
                for ev in eval_rows
                if ev["id"] == sources[0][1]
            }
            self.assertEqual(
                roles_for_first,
                {"pair_anchor_a", "pair_anchor_b", "generated_direct"},
            )

    def test_no_evaluations_flag_skips_eval_file(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            (run_dir / "examples").mkdir(parents=True)
            doc = _example_doc(source="med", source_id="0006001-aaaaaa", pair_label="b")
            (run_dir / "examples" / "med__0006001-aaaaaa.json").write_text(
                json.dumps(doc), encoding="utf-8"
            )

            out_dir = tmp_path / "out"
            rc = _pkg.main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--out-dir",
                    str(out_dir),
                    "--stem",
                    "smoke",
                    "--no-evaluations",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "smoke_rubrics.jsonl").exists())
            self.assertFalse((out_dir / "smoke_rubric_evaluations.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
