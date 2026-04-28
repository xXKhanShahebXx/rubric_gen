import unittest
from unittest.mock import patch

from rubric_gen.compiled.discovery import (
    RecursiveDiscoveryConfig,
    discover_pair_criteria,
    filter_pair_grounded_proposals,
    pair_id,
)
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec


def _candidate(
    *,
    candidate_id: str,
    text: str,
    source_label: str,
    origin_kind: str,
    mutation_id: str | None = None,
) -> CandidateNote:
    metadata = {}
    if mutation_id:
        metadata["mutation_id"] = mutation_id
    return CandidateNote(
        candidate_id=candidate_id,
        example_id="ex_1",
        text=text,
        source_label=source_label,
        quality_bucket="test",
        origin_kind=origin_kind,
        metadata=metadata,
    )


def _example_record() -> ExampleRecord:
    return ExampleRecord(
        example_id="ex_1",
        source="test",
        source_id="ex_1",
        dataset_subset="unit",
        conversation="Patient has abdominal pain. Include follow-up timing and return precautions.",
        task_prompt="Write a clinically faithful medical note from the source transcript.",
        reference_note="Plan: Follow up in 48 hours if not improving. Return sooner for worsening pain.",
        task_profile_id="note_documentation",
        task_family_id="general_clinical_note",
        artifact_kind="note",
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        alias="test",
        provider="test",
        model="test-model",
        api_key_env="TEST_API_KEY",
    )


class PairGroundingFilterTests(unittest.TestCase):
    def test_flatten_structure_rejects_off_target_content_criteria(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "CHIEF COMPLAINT\n"
                "Abdominal pain.\n\n"
                "ASSESSMENT AND PLAN\n"
                "- Medical Reasoning: Based on symptoms, exam findings, and ultrasound, this is mild cholecystitis.\n"
                "- Patient Education and Counseling: We discussed surgery and follow-up timing.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:flatten_structure",
            origin_kind="synthetic_mutation",
            mutation_id="flatten_structure",
            text=(
                "Abdominal pain.\n\n"
                "- Medical Reasoning: Based on symptoms, exam findings, and ultrasound, this is mild cholecystitis.\n"
                "- Patient Education and Counseling: We discussed surgery and follow-up timing.\n"
            ),
        )
        proposals = [
            {
                "dimension": "structure",
                "label": "Chief complaint section present",
                "requirement": "The note includes a distinct Chief Complaint section explicitly stating the patient's main reason for visit.",
                "severity_tier": "medium",
                "rationale": "This preserves the expected structure.",
            },
            {
                "dimension": "diagnostic_reasoning",
                "label": "Explicit medical reasoning for assessment",
                "requirement": "The note contains a clear medical reasoning statement linking symptoms, exam findings, and imaging to the diagnosis.",
                "severity_tier": "high",
                "rationale": "This captures clinical thinking.",
            },
            {
                "dimension": "patient_education",
                "label": "Detailed patient education and counseling documented",
                "requirement": "The note explicitly documents a thorough discussion with the patient about the surgical treatment, including follow-up timing.",
                "severity_tier": "high",
                "rationale": "This captures patient counseling.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Chief complaint section present"],
        )
        self.assertEqual(len(rejected), 2)
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 2)

    def test_inflate_certainty_accepts_certainty_aligned_proposals(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text="Assessment: Likely viral illness. Plan: Supportive care and return precautions.",
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:inflate_certainty",
            origin_kind="synthetic_mutation",
            mutation_id="inflate_certainty",
            text=(
                "Assessment: Likely viral illness. Plan: Supportive care and return precautions.\n\n"
                "Addendum: Findings are pathognomonic for the working diagnosis per clinical assessment."
            ),
        )
        proposals = [
            {
                "dimension": "certainty_language",
                "label": "Avoid unsupported diagnostic certainty",
                "requirement": "The note avoids definitive or pathognomonic diagnostic language when support is limited.",
                "severity_tier": "hard_gate",
                "rationale": "This prevents overstating the diagnosis.",
            }
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(grounding["delta_mode"], "weak_only")

    def test_strip_symptom_detail_lines_rejects_exam_and_negative_noise(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Constitutional: Reports low-grade fevers.\n"
                "Body Temperature: Afebrile.\n"
                "- Examination of Abdomen: Soft, nondistended abdomen. Positive slight guarding to the right upper quadrant, but without rebound tenderness. Positive for Murphy signs. Peritoneal signs not appreciated.\n"
                "- Auscultation: Bowel sounds normal in all 4 quadrants.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:strip_symptom_detail_lines",
            origin_kind="synthetic_mutation",
            mutation_id="strip_symptom_detail_lines",
            text="",
        )
        proposals = [
            {
                "dimension": "symptom_detail",
                "label": "Document low-grade fevers in review of systems",
                "requirement": "The note must document the presence or absence of low-grade fevers in the review of systems.",
                "severity_tier": "high",
                "rationale": "Fever detail matters here.",
            },
            {
                "dimension": "symptom_detail",
                "label": "Include pain characteristics in review of systems",
                "requirement": "The note must include details about the location and nature of the abdominal pain in the review of systems.",
                "severity_tier": "medium",
                "rationale": "Pain characterization matters here.",
            },
            {
                "dimension": "symptom_detail",
                "label": "Document nausea and vomiting status",
                "requirement": "The note must specify that the patient reports nausea but denies vomiting.",
                "severity_tier": "high",
                "rationale": "Associated symptoms matter.",
            },
            {
                "dimension": "physical_exam",
                "label": "Document bowel sounds in all four quadrants",
                "requirement": "The note must document auscultation of bowel sounds in all four abdominal quadrants.",
                "severity_tier": "medium",
                "rationale": "Exam detail matters.",
            },
            {
                "dimension": "vital_signs",
                "label": "Document all relevant vital signs including oxygen saturation",
                "requirement": "The note must document all vital signs including blood pressure, heart rate, respiratory rate, oxygen saturation, and temperature or afebrile status.",
                "severity_tier": "medium",
                "rationale": "Broader vital-sign documentation matters.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            [
                "Document low-grade fevers in review of systems",
                "Include pain characteristics in review of systems",
            ],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document nausea and vomiting status",
                "Document bowel sounds in all four quadrants",
                "Document all relevant vital signs including oxygen saturation",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 2)
        self.assertEqual(grounding["rejected_count"], 3)

    def test_drop_study_mentions_rejects_plan_noise(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Abdominal ultrasound obtained at an outside facility is reviewed today. "
                "This demonstrates multiple gallstones and mild gallbladder thickening.\n"
                "- Medical Reasoning: Based on symptoms, exam findings, and ultrasound, this is mild cholecystitis.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_study_mentions",
            origin_kind="synthetic_mutation",
            mutation_id="drop_study_mentions",
            text="",
        )
        proposals = [
            {
                "dimension": "test_mentions",
                "label": "Include ultrasound results",
                "requirement": "The note must document the abdominal ultrasound findings including gallstones and gallbladder thickening.",
                "severity_tier": "high",
                "rationale": "Study results matter here.",
            },
            {
                "dimension": "diagnostic_reasoning",
                "label": "Explain diagnosis with study correlation",
                "requirement": "The note must link symptoms, exam findings, and imaging to the diagnosis.",
                "severity_tier": "medium",
                "rationale": "Study-linked reasoning matters here.",
            },
            {
                "dimension": "assessment_and_plan",
                "label": "Include plan for diabetes management follow-up",
                "requirement": "The note must document recommendations for diabetes follow-up in context of upcoming surgery.",
                "severity_tier": "low",
                "rationale": "Comorbidity planning matters here.",
            },
            {
                "dimension": "patient_education",
                "label": "Document detailed patient education about surgery",
                "requirement": "The note must describe the preoperative and postoperative course of care discussed with the patient.",
                "severity_tier": "medium",
                "rationale": "Surgery counseling matters here.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            [
                "Include ultrasound results",
                "Explain diagnosis with study correlation",
            ],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Include plan for diabetes management follow-up",
                "Document detailed patient education about surgery",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 2)
        self.assertEqual(grounding["rejected_count"], 2)

    def test_drop_followup_lines_rejects_return_and_medication_noise(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Plan: Follow up in 6 weeks with repeat clinic evaluation.\n"
                "Instructions: Record daily blood pressure readings and upload them to the patient portal.\n"
                "Return precautions: Return sooner for worsening pain or fever.\n"
                "Treatment: Continue ibuprofen as needed.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_followup_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_followup_lines",
            text=(
                "Instructions: Record daily blood pressure readings and upload them to the patient portal.\n"
                "Treatment: Continue ibuprofen as needed.\n"
            ),
        )
        proposals = [
            {
                "dimension": "follow_up_specificity",
                "label": "Document follow-up interval",
                "requirement": "The note must document planned outpatient follow-up timing and recheck interval when discussed.",
                "severity_tier": "high",
                "rationale": "This preserves scheduled follow-up.",
            },
            {
                "dimension": "return_precautions",
                "label": "Document return precautions",
                "requirement": "The note must tell the patient when to return sooner for worsening pain or fever.",
                "severity_tier": "medium",
                "rationale": "This preserves escalation guidance.",
            },
            {
                "dimension": "medication_management",
                "label": "Document medication plan",
                "requirement": "The note must document the ibuprofen treatment plan.",
                "severity_tier": "medium",
                "rationale": "This preserves treatment details.",
            },
            {
                "dimension": "follow_up_specificity",
                "label": "Document home blood pressure monitoring",
                "requirement": "The note must document daily blood pressure monitoring and portal reporting instructions.",
                "severity_tier": "medium",
                "rationale": "This preserves monitoring guidance.",
            },
            {
                "dimension": "follow_up_specificity",
                "label": "Document patient agreement with follow-up plan",
                "requirement": "The note must state that the patient understands and agrees with the follow-up plan.",
                "severity_tier": "low",
                "rationale": "This preserves agreement language.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Document follow-up interval"],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document return precautions",
                "Document medication plan",
                "Document home blood pressure monitoring",
                "Document patient agreement with follow-up plan",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 4)

    def test_drop_return_precaution_lines_accepts_precautions_only(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Plan: Follow up in 2 weeks for reassessment.\n"
                "Return precautions: Seek urgent care for worsening shortness of breath.\n"
                "Medication: Continue albuterol inhaler as needed.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_return_precaution_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_return_precaution_lines",
            text=(
                "Plan: Follow up in 2 weeks for reassessment.\n"
                "Medication: Continue albuterol inhaler as needed.\n"
            ),
        )
        proposals = [
            {
                "dimension": "return_precautions",
                "label": "Document escalation guidance",
                "requirement": "The note must include return precautions or urgent-care guidance for worsening shortness of breath.",
                "severity_tier": "high",
                "rationale": "This preserves safety guidance.",
            },
            {
                "dimension": "follow_up_specificity",
                "label": "Document follow-up appointment timing",
                "requirement": "The note must document the follow-up appointment timing.",
                "severity_tier": "medium",
                "rationale": "This preserves routine follow-up.",
            },
            {
                "dimension": "medication_management",
                "label": "Document inhaler regimen",
                "requirement": "The note must document the albuterol inhaler plan.",
                "severity_tier": "medium",
                "rationale": "This preserves medication details.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Document escalation guidance"],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document follow-up appointment timing",
                "Document inhaler regimen",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 2)

    def test_drop_medication_lines_rejects_intervention_and_followup_noise(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Medication: Start albuterol inhaler 2 puffs every 4 hours as needed.\n"
                "Plan: Continue to monitor blood pressure at home and upload readings.\n"
                "Plan: Refer to physical therapy.\n"
                "Plan: Follow up in 2 weeks.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_medication_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_medication_lines",
            text=(
                "Plan: Continue to monitor blood pressure at home and upload readings.\n"
                "Plan: Refer to physical therapy.\n"
                "Plan: Follow up in 2 weeks.\n"
            ),
        )
        proposals = [
            {
                "dimension": "medication_management",
                "label": "Document medication name and dosing",
                "requirement": "The note must document the inhaler medication, dose, and as-needed frequency when discussed.",
                "severity_tier": "high",
                "rationale": "This preserves medication details.",
            },
            {
                "dimension": "intervention_plan",
                "label": "Document physical therapy referral",
                "requirement": "The note must document the physical therapy referral plan.",
                "severity_tier": "medium",
                "rationale": "This preserves referral planning.",
            },
            {
                "dimension": "follow_up_specificity",
                "label": "Document follow-up interval",
                "requirement": "The note must document outpatient follow-up timing.",
                "severity_tier": "medium",
                "rationale": "This preserves follow-up planning.",
            },
            {
                "dimension": "medication_management",
                "label": "Document home blood pressure monitoring plan",
                "requirement": "The note must include home blood pressure monitoring instructions.",
                "severity_tier": "medium",
                "rationale": "This preserves monitoring guidance.",
            },
            {
                "dimension": "medication_management",
                "label": "Document patient agreement with medication plan",
                "requirement": "The note must state that the patient understands and agrees with the medication plan.",
                "severity_tier": "low",
                "rationale": "This preserves agreement language.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Document medication name and dosing"],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document physical therapy referral",
                "Document follow-up interval",
                "Document home blood pressure monitoring plan",
                "Document patient agreement with medication plan",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 4)

    def test_drop_procedure_lines_rejects_medication_and_testing_noise(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Plan: Recommend endoscopy referral if symptoms persist.\n"
                "Discussion: Diagnosis and treatment options were reviewed with the patient.\n"
                "Instructions: Avoid hiking until pain improves.\n"
                "Medication: Continue protonix daily.\n"
                "Testing: Repeat CBC in 1 week.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_procedure_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_procedure_lines",
            text=(
                "Discussion: Diagnosis and treatment options were reviewed with the patient.\n"
                "Instructions: Avoid hiking until pain improves.\n"
                "Medication: Continue protonix daily.\n"
                "Testing: Repeat CBC in 1 week.\n"
            ),
        )
        proposals = [
            {
                "dimension": "intervention_plan",
                "label": "Document endoscopy or referral plan",
                "requirement": "The note must document recommended endoscopy or specialty referral when discussed.",
                "severity_tier": "high",
                "rationale": "This preserves intervention planning.",
            },
            {
                "dimension": "medication_management",
                "label": "Document daily protonix regimen",
                "requirement": "The note must document the daily protonix medication plan.",
                "severity_tier": "medium",
                "rationale": "This preserves medication detail.",
            },
            {
                "dimension": "testing_plan",
                "label": "Document repeat CBC plan",
                "requirement": "The note must document the repeat CBC testing plan.",
                "severity_tier": "medium",
                "rationale": "This preserves planned workup.",
            },
            {
                "dimension": "intervention_plan",
                "label": "Document treatment discussion with patient",
                "requirement": "The note must document that diagnosis and treatment options were discussed with the patient.",
                "severity_tier": "low",
                "rationale": "This preserves discussion detail.",
            },
            {
                "dimension": "intervention_plan",
                "label": "Document activity restrictions",
                "requirement": "The note must tell the patient to avoid hiking until pain improves.",
                "severity_tier": "medium",
                "rationale": "This preserves activity guidance.",
            },
            {
                "dimension": "intervention_plan",
                "label": "Recommend home blood pressure cuff purchase",
                "requirement": "The note must tell the patient to obtain a home blood pressure cuff for monitoring.",
                "severity_tier": "medium",
                "rationale": "This preserves home monitoring equipment guidance.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Document endoscopy or referral plan"],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document daily protonix regimen",
                "Document repeat CBC plan",
                "Document treatment discussion with patient",
                "Document activity restrictions",
                "Recommend home blood pressure cuff purchase",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 5)

    def test_drop_testing_plan_lines_accepts_workup_only(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Plan: Order repeat CBC and abdominal ultrasound next week.\n"
                "Results: Outside-facility ultrasound findings reviewed today show gallstones.\n"
                "Medication: Continue ibuprofen as needed.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_testing_plan_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_testing_plan_lines",
            text=(
                "Results: Outside-facility ultrasound findings reviewed today show gallstones.\n"
                "Medication: Continue ibuprofen as needed.\n"
            ),
        )
        proposals = [
            {
                "dimension": "testing_plan",
                "label": "Document ordered repeat CBC and ultrasound",
                "requirement": "The note must document the ordered repeat CBC and ultrasound workup plan.",
                "severity_tier": "high",
                "rationale": "This preserves planned testing.",
            },
            {
                "dimension": "study_results",
                "label": "Document reviewed ultrasound findings",
                "requirement": "The note must document outside-facility ultrasound findings and reviewed results.",
                "severity_tier": "medium",
                "rationale": "This preserves past study findings.",
            },
            {
                "dimension": "medication_management",
                "label": "Document ibuprofen regimen",
                "requirement": "The note must document the ibuprofen medication plan.",
                "severity_tier": "medium",
                "rationale": "This preserves medication detail.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Document ordered repeat CBC and ultrasound"],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document reviewed ultrasound findings",
                "Document ibuprofen regimen",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 2)

    def test_drop_assessment_reasoning_lines_accepts_reasoning_only(self) -> None:
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Assessment: Findings are consistent with acute cholecystitis based on right upper quadrant pain and ultrasound findings.\n"
                "Plan: We recommend surgery due to persistent symptoms.\n"
                "Follow-up: Return to clinic in 1 week.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_assessment_reasoning_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_assessment_reasoning_lines",
            text="Follow-up: Return to clinic in 1 week.\n",
        )
        proposals = [
            {
                "dimension": "diagnostic_reasoning",
                "label": "Link assessment and plan to supporting evidence",
                "requirement": "The note must document the rationale linking symptoms and ultrasound findings to the assessment and chosen plan.",
                "severity_tier": "high",
                "rationale": "This captures assessment reasoning.",
            },
            {
                "dimension": "follow_up_specificity",
                "label": "Document return-to-clinic timing",
                "requirement": "The note must document return-to-clinic follow-up timing.",
                "severity_tier": "medium",
                "rationale": "This captures follow-up timing.",
            },
            {
                "dimension": "diagnostic_reasoning",
                "label": "Explain why patient should call if symptoms worsen",
                "requirement": "The note must explain why the patient should call the clinic if symptoms worsen.",
                "severity_tier": "low",
                "rationale": "This captures instruction reasoning.",
            },
        ]

        accepted, rejected, grounding = filter_pair_grounded_proposals(
            strong=strong,
            weak=weak,
            proposals=proposals,
        )

        self.assertEqual(
            [row["label"] for row in accepted],
            ["Link assessment and plan to supporting evidence"],
        )
        self.assertEqual(
            [row["label"] for row in rejected],
            [
                "Document return-to-clinic timing",
                "Explain why patient should call if symptoms worsen",
            ],
        )
        self.assertEqual(grounding["accepted_count"], 1)
        self.assertEqual(grounding["rejected_count"], 2)


class RecursivePairDiscoveryTests(unittest.TestCase):
    def test_recursive_discovery_replaces_broad_parent_with_grounded_child(self) -> None:
        example = _example_record()
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text=(
                "Plan: Follow up in 48 hours if not improving.\n"
                "Return sooner for worsening pain.\n"
            ),
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_followup_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_followup_lines",
            text="Return sooner for worsening pain.\n",
        )
        pair_key = pair_id(strong, weak)
        root_row = {
            "dimension": "content_coverage",
            "label": "Main next steps covered",
            "requirement": "The note includes follow-up timing and the main next-step guidance.",
            "severity_tier": "high",
            "rationale": "This broadly captures the missing plan details.",
            "example_id": example.example_id,
            "pair_id": pair_key,
            "strong_candidate_id": strong.candidate_id,
            "weak_candidate_id": weak.candidate_id,
            "proposal_index": 0,
            "task_profile_id": "note_documentation",
        }
        good_child = {
            "dimension": "follow_up_specificity",
            "label": "Document follow-up interval",
            "requirement": "The note specifies a follow-up interval such as returning in 48 hours if not improving.",
            "severity_tier": "high",
            "rationale": "This preserves the explicit follow-up timing.",
            "example_id": example.example_id,
            "pair_id": pair_key,
            "strong_candidate_id": strong.candidate_id,
            "weak_candidate_id": weak.candidate_id,
            "proposal_index": 0,
            "task_profile_id": "note_documentation",
        }
        off_target_child = {
            "dimension": "style",
            "label": "Use concise grammar",
            "requirement": "The note uses concise grammar and polished sentence structure.",
            "severity_tier": "low",
            "rationale": "This is off target for the mutation.",
            "example_id": example.example_id,
            "pair_id": pair_key,
            "strong_candidate_id": strong.candidate_id,
            "weak_candidate_id": weak.candidate_id,
            "proposal_index": 1,
            "task_profile_id": "note_documentation",
        }

        with patch(
            "rubric_gen.compiled.discovery.propose_criteria_for_pair",
            return_value=([root_row], False, None),
        ), patch(
            "rubric_gen.compiled.discovery._decompose_criterion_for_pair",
            return_value=([good_child, off_target_child], False, None),
        ):
            result = discover_pair_criteria(
                example=example,
                strong=strong,
                weak=weak,
                model_spec=_model_spec(),
                router=object(),  # type: ignore[arg-type]
                cache=None,
                max_criteria=4,
                task_profile_id="note_documentation",
                artifact_label="note",
                recursive_config=RecursiveDiscoveryConfig(
                    enabled=True,
                    max_depth=1,
                    max_recursive_parents_per_pair=1,
                    max_children_per_parent=3,
                    max_recursive_calls_per_pair=1,
                ),
            )

        self.assertEqual(
            [row["label"] for row in result["proposals"]],
            ["Document follow-up interval"],
        )
        self.assertEqual(result["recursion"]["recursive_calls"], 1)
        self.assertEqual(result["recursion"]["recursive_children_promoted"], 1)
        self.assertTrue(result["recursion"]["changed_structure"])
        child = result["proposals"][0]
        self.assertEqual(child["recursion_depth"], 1)
        self.assertEqual(child["decomposition_source"], "criterion_decomposition")
        self.assertEqual(child["parent_criterion_id"], result["promoted_root_proposals"][0]["criterion_id"])
        self.assertEqual(child["root_pair_id"], pair_key)
        self.assertIn("Use concise grammar", [row["label"] for row in result["rejected_proposals"]])

    def test_recursive_discovery_keeps_parent_when_disabled(self) -> None:
        example = _example_record()
        strong = _candidate(
            candidate_id="strong",
            source_label="reference_note",
            origin_kind="original",
            text="Plan: Follow up in 48 hours if not improving.\n",
        )
        weak = _candidate(
            candidate_id="weak",
            source_label="synthetic_mutation:drop_followup_lines",
            origin_kind="synthetic_mutation",
            mutation_id="drop_followup_lines",
            text="Plan: Return sooner for worsening pain.\n",
        )
        pair_key = pair_id(strong, weak)
        root_row = {
            "dimension": "content_coverage",
            "label": "Main next steps covered",
            "requirement": "The note includes follow-up timing and the main next-step guidance.",
            "severity_tier": "high",
            "rationale": "This broadly captures the missing plan details.",
            "example_id": example.example_id,
            "pair_id": pair_key,
            "strong_candidate_id": strong.candidate_id,
            "weak_candidate_id": weak.candidate_id,
            "proposal_index": 0,
            "task_profile_id": "note_documentation",
        }

        with patch(
            "rubric_gen.compiled.discovery.propose_criteria_for_pair",
            return_value=([root_row], False, None),
        ), patch("rubric_gen.compiled.discovery._decompose_criterion_for_pair") as mock_decompose:
            result = discover_pair_criteria(
                example=example,
                strong=strong,
                weak=weak,
                model_spec=_model_spec(),
                router=object(),  # type: ignore[arg-type]
                cache=None,
                max_criteria=4,
                task_profile_id="note_documentation",
                artifact_label="note",
                recursive_config=RecursiveDiscoveryConfig(enabled=False),
            )

        mock_decompose.assert_not_called()
        self.assertEqual(
            [row["label"] for row in result["proposals"]],
            ["Main next steps covered"],
        )
        self.assertEqual(result["recursion"]["recursive_calls"], 0)
        self.assertFalse(result["recursion"]["changed_structure"])
        self.assertEqual(result["proposals"][0]["decomposition_source"], "root_pair_discovery")


if __name__ == "__main__":
    unittest.main()
