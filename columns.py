from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@dataclass
class Cols:
    user: str
    item: str
    skill: str
    correct: str
    order: str
    hint_count: Optional[str]
    first_action: Optional[str]
    bottom_hint: Optional[str]
    opportunity: Optional[str]


def infer_cols(df: pd.DataFrame) -> Cols:
    user = first_existing(df, ["user_id", "student_id", "studentId", "uid", "user", "Anon Student Id"])
    item = first_existing(
        df,
        [
            "problem_id",
            "problemId",
            "item_id",
            "question_id",
            "problem",
            "exer_id",
            "Problem Name",
            "Step Name",
        ],
    )
    skill = first_existing(
        df,
        [
            "skill_id",
            "kc_id",
            "knowledge_id",
            "skill",
            "kc",
            "concept_id",
            "skill_name",
            "KC(Default)",
            "tags",
        ],
    )
    correct = first_existing(df, ["correct", "is_correct", "label", "Correct First Attempt"])
    order = first_existing(
        df,
        [
            "order_id",
            "timestamp",
            "time",
            "start_time",
            "problem_log_id",
            "action_num",
            "actionId",
            "startTime",
            "endTime",
            "Step Start Time",
            "First Transaction Time",
        ],
    )
    hint_count = first_existing(
        df,
        [
            "hint_count",
            "hints_used",
            "hint_used",
            "num_hints",
            "hintCount",
            "hint",
            "frIsHelpRequest",
            "stlHintUsed",
            "Hints",
        ],
    )
    first_action = first_existing(df, ["first_action", "firstaction"])
    bottom_hint = first_existing(df, ["bottom_hint", "bottomout_hint", "bottom_out_hint", "bottomHint"])
    opportunity = first_existing(
        df,
        [
            "opportunity",
            "opp",
            "skill_opportunity",
            "totalFrSkillOpportunities",
            "frTotalSkillOpportunitiesScaffolding",
            "totalFrSkillOpportunitiesByScaffolding",
            "Opportunity(Default)",
        ],
    )

    missing = [("user", user), ("item", item), ("skill", skill), ("correct", correct)]
    miss = [k for k, v in missing if v is None]
    if miss:
        raise ValueError(f"Missing required columns: {miss}. Available: {list(df.columns)[:50]} ...")
    if order is None:
        # fallback: create a pseudo order
        df["_row_order"] = np.arange(len(df))
        order = "_row_order"
    return Cols(
        user=user,
        item=item,
        skill=skill,
        correct=correct,
        order=order,
        hint_count=hint_count,
        first_action=first_action,
        bottom_hint=bottom_hint,
        opportunity=opportunity,
    )


def build_treatment_A(df: pd.DataFrame, cols: Cols) -> np.ndarray:
    A = np.zeros(len(df), dtype=np.int64)
    if cols.hint_count is not None:
        hc = pd.to_numeric(df[cols.hint_count], errors="coerce").fillna(0.0).values
        A = np.maximum(A, (hc > 0).astype(np.int64))
    if cols.first_action is not None:
        fa = df[cols.first_action].astype(str).str.lower().values
        A = np.maximum(A, np.array([1 if ("hint" in s) else 0 for s in fa], dtype=np.int64))
    if cols.bottom_hint is not None:
        bh = pd.to_numeric(df[cols.bottom_hint], errors="coerce").fillna(0.0).values
        A = np.maximum(A, (bh > 0).astype(np.int64))
    return A
