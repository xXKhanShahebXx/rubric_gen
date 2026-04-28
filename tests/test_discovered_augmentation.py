import unittest

from rubric_gen.compiled.discovered_augmentation import (
    criterion_templates_from_merged_proposals,
    map_discovery_dimension_to_ontology,
)


class DiscoveredAugmentationTests(unittest.TestCase):
    def test_dimension_mapping_handles_new_management_dimensions(self) -> None:
        self.assertEqual(
            map_discovery_dimension_to_ontology("follow_up_specificity"),
            ("management_plan", "follow_up_specificity"),
        )
        self.assertEqual(
            map_discovery_dimension_to_ontology("return_precautions"),
            ("management_plan", "return_precautions"),
        )
        self.assertEqual(
            map_discovery_dimension_to_ontology("medication_management"),
            ("management_plan", "medication_management"),
        )
        self.assertEqual(
            map_discovery_dimension_to_ontology("intervention_plan"),
            ("management_plan", "intervention_plan"),
        )
        self.assertEqual(
            map_discovery_dimension_to_ontology("testing_plan"),
            ("management_plan", "testing_plan"),
        )
        self.assertEqual(
            map_discovery_dimension_to_ontology("treatment_grounding"),
            ("management_plan", "treatment_grounding"),
        )
        self.assertEqual(
            map_discovery_dimension_to_ontology("diagnostic_reasoning"),
            ("diagnostic_reasoning", "assessment_linkage"),
        )

    def test_structure_section_rows_collapse_to_one_template(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "structure||high||clear headers||distinct section headers",
                    "dimension": "structure",
                    "label": "Presence of clear section headers",
                    "requirement": "The note must include distinct section headers such as CHIEF COMPLAINT and ASSESSMENT AND PLAN.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["pair_1"],
                },
                {
                    "merge_key": "assessment_and_plan_detail||medium||subsections||assessment and plan subsections",
                    "dimension": "assessment_and_plan_detail",
                    "label": "Separate assessment and plan subsections",
                    "requirement": "The note must separate assessment, patient education, and medical treatment into distinct subsections under the plan.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["pair_1"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={"ex_1": "soap_note"},
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["label"],
            "Expected note-family section scaffold present",
        )
        self.assertEqual(selected[0]["severity_tier_discovery"], "medium")
        self.assertEqual(selected[0]["note_family_scope"], ["soap_note"])
        self.assertEqual(
            len(selected[0]["consolidated_from_merge_keys"]),
            2,
        )

    def test_strip_symptom_detail_rows_collapse_to_one_template(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "symptom_detail||high||pain||pain characteristics",
                    "dimension": "symptom_detail",
                    "label": "Include pain characteristics in review of systems",
                    "requirement": "The note must include details about the location and nature of the abdominal pain in the review of systems.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__strip_symptom_detail_lines"],
                },
                {
                    "merge_key": "vital_signs||medium||temp||temperature status",
                    "dimension": "vital_signs",
                    "label": "Include body temperature or afebrile status in vitals",
                    "requirement": "The note must include the patient's body temperature or explicitly state afebrile status in the vital signs section.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__strip_symptom_detail_lines"],
                },
                {
                    "merge_key": "physical_exam||medium||peritoneal||peritoneal signs",
                    "dimension": "physical_exam",
                    "label": "Document absence of peritoneal signs",
                    "requirement": "The note must explicitly state whether peritoneal signs are present or absent.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__strip_symptom_detail_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={"ex_1": "soap_note"},
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["label"],
            "Salient encounter symptom detail preserved",
        )
        self.assertEqual(selected[0]["severity_tier_discovery"], "medium")
        self.assertEqual(selected[0]["note_family_scope"], ["soap_note"])
        self.assertEqual(selected[0]["consolidation_kind"], "symptom_detail_union")
        self.assertEqual(
            len(selected[0]["consolidated_from_merge_keys"]),
            3,
        )

    def test_provisional_discovered_hard_gate_is_capped(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "test_mentions||hard_gate||include ultrasound findings||ultrasound findings required",
                    "dimension": "test_mentions",
                    "label": "Include ultrasound findings",
                    "requirement": "The note must include ultrasound findings supporting the diagnosis.",
                    "severity_tier": "hard_gate",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_study_mentions"],
                }
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={"ex_1": "soap_note"},
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0].severity_tier, "essential")
        self.assertEqual(selected[0]["severity_tier_ontology_raw"], "catastrophic")
        self.assertEqual(selected[0]["severity_tier_ontology"], "essential")
        self.assertTrue(selected[0]["severity_tier_capped"])

    def test_inflate_certainty_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "certainty_language||high||avoid unsupported certainty||avoid unsupported certainty",
                    "dimension": "certainty_language",
                    "label": "Avoid unsupported certainty statements",
                    "requirement": "The note should not include definitive or pathognomonic statements about diagnosis without explicit clinical evidence.",
                    "severity_tier": "high",
                    "count": 2,
                    "example_ids": ["ex_1", "ex_2"],
                    "pair_ids": [
                        "ex_1__reference_note__vs__ex_1__mut__inflate_certainty",
                        "ex_2__reference_note__vs__ex_2__mut__inflate_certainty",
                    ],
                },
                {
                    "merge_key": "diagnostic_reasoning||medium||include uncertainty||include diagnostic uncertainty",
                    "dimension": "diagnostic_reasoning",
                    "label": "Include appropriate diagnostic uncertainty",
                    "requirement": "The note should reflect diagnostic uncertainty or conditional language when the dialogue indicates ongoing assessment.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_3"],
                    "pair_ids": [
                        "ex_3__reference_note__vs__ex_3__mut__inflate_certainty",
                    ],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "general_clinical_note",
                "ex_2": "general_clinical_note",
                "ex_3": "general_clinical_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["label"],
            "Appropriate diagnostic certainty language",
        )
        self.assertEqual(selected[0]["consolidation_kind"], "certainty_language_union")
        self.assertEqual(selected[0]["note_family_scope"], ["general_clinical_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)

    def test_followup_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "follow_up_specificity||high||timing||follow-up timing",
                    "dimension": "follow_up_specificity",
                    "label": "Document follow-up timing",
                    "requirement": "The note must document outpatient follow-up timing and planned recheck interval.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_followup_lines"],
                },
                {
                    "merge_key": "management_plan||medium||reassess||planned reevaluation",
                    "dimension": "management_plan",
                    "label": "Include planned reevaluation",
                    "requirement": "The note must document planned reevaluation or reassessment at follow-up when discussed.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_2"],
                    "pair_ids": ["ex_2__reference_note__vs__ex_2__mut__drop_followup_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "general_clinical_note",
                "ex_2": "general_clinical_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], "Specific follow-up timing documented")
        self.assertEqual(selected[0]["consolidation_kind"], "follow_up_timing_union")
        self.assertEqual(selected[0]["note_family_scope"], ["general_clinical_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)

    def test_return_precaution_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "return_precautions||high||worsen||worsening symptoms",
                    "dimension": "return_precautions",
                    "label": "Document worsening-symptom precautions",
                    "requirement": "The note must document return precautions for worsening symptoms.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_return_precaution_lines"],
                },
                {
                    "merge_key": "management_plan||medium||urgent||urgent care guidance",
                    "dimension": "management_plan",
                    "label": "Include urgent-care escalation guidance",
                    "requirement": "The note must tell the patient when to seek urgent care or call the office.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_2"],
                    "pair_ids": ["ex_2__reference_note__vs__ex_2__mut__drop_return_precaution_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "soap_note",
                "ex_2": "soap_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], "Clear return precautions documented")
        self.assertEqual(selected[0]["consolidation_kind"], "return_precautions_union")
        self.assertEqual(selected[0]["note_family_scope"], ["soap_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)

    def test_medication_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "medication_management||high||meds||medication details",
                    "dimension": "medication_management",
                    "label": "Document anticoagulant therapy",
                    "requirement": "The note must document anticoagulant medication and dosing details.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_medication_lines"],
                },
                {
                    "merge_key": "management_plan||medium||refill||refill plan",
                    "dimension": "management_plan",
                    "label": "Include refill or continuation details",
                    "requirement": "The note must document refill or continuation instructions for the medication plan.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_2"],
                    "pair_ids": ["ex_2__reference_note__vs__ex_2__mut__drop_medication_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "soap_note",
                "ex_2": "soap_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], "Grounded medication plan documented")
        self.assertEqual(selected[0]["consolidation_kind"], "medication_management_union")
        self.assertEqual(selected[0]["note_family_scope"], ["soap_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)

    def test_intervention_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "intervention_plan||high||referral||endoscopy referral",
                    "dimension": "intervention_plan",
                    "label": "Document specialty referral",
                    "requirement": "The note must document the endoscopy or specialty referral plan.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_procedure_lines"],
                },
                {
                    "merge_key": "management_plan||medium||brace||device plan",
                    "dimension": "management_plan",
                    "label": "Include brace or device recommendation",
                    "requirement": "The note must document a brace, splint, or other device recommendation when discussed.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_2"],
                    "pair_ids": ["ex_2__reference_note__vs__ex_2__mut__drop_procedure_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "general_clinical_note",
                "ex_2": "general_clinical_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], "Grounded intervention plan documented")
        self.assertEqual(selected[0]["consolidation_kind"], "intervention_plan_union")
        self.assertEqual(selected[0]["note_family_scope"], ["general_clinical_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)

    def test_testing_plan_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "testing_plan||high||cbc||repeat cbc",
                    "dimension": "testing_plan",
                    "label": "Document repeat CBC plan",
                    "requirement": "The note must document the repeat CBC testing plan.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_testing_plan_lines"],
                },
                {
                    "merge_key": "management_plan||medium||ultrasound||ordered ultrasound",
                    "dimension": "management_plan",
                    "label": "Include ordered ultrasound workup",
                    "requirement": "The note must document ordered ultrasound or imaging follow-through when discussed.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_2"],
                    "pair_ids": ["ex_2__reference_note__vs__ex_2__mut__drop_testing_plan_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "general_clinical_note",
                "ex_2": "general_clinical_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], "Planned diagnostic testing documented")
        self.assertEqual(selected[0]["consolidation_kind"], "testing_plan_union")
        self.assertEqual(selected[0]["note_family_scope"], ["general_clinical_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)

    def test_reasoning_rows_collapse_per_note_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "diagnostic_reasoning||high||linkage||assessment linkage",
                    "dimension": "diagnostic_reasoning",
                    "label": "Link diagnosis to supporting symptoms",
                    "requirement": "The note must explain the rationale linking symptoms and imaging to the working diagnosis.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["ex_1"],
                    "pair_ids": ["ex_1__reference_note__vs__ex_1__mut__drop_assessment_reasoning_lines"],
                },
                {
                    "merge_key": "assessment_rationale||medium||plan choice||plan choice rationale",
                    "dimension": "assessment_rationale",
                    "label": "Explain why the selected intervention was chosen",
                    "requirement": "The note must document why the selected intervention was chosen based on risk and reviewed results.",
                    "severity_tier": "medium",
                    "count": 1,
                    "example_ids": ["ex_2"],
                    "pair_ids": ["ex_2__reference_note__vs__ex_2__mut__drop_assessment_reasoning_lines"],
                },
            ]
        }

        templates, selected = criterion_templates_from_merged_proposals(
            merged,
            example_id_to_nf={
                "ex_1": "general_clinical_note",
                "ex_2": "general_clinical_note",
            },
            support_threshold=1,
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], "Assessment linked to supporting evidence")
        self.assertEqual(selected[0]["consolidation_kind"], "diagnostic_reasoning_union")
        self.assertEqual(selected[0]["note_family_scope"], ["general_clinical_note"])
        self.assertEqual(len(selected[0]["consolidated_from_merge_keys"]), 2)


if __name__ == "__main__":
    unittest.main()
