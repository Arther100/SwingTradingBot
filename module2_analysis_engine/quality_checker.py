"""
SwingAdvisorBot — Module 2: AI Analysis Engine
quality_checker.py — Self-reflection quality gate for Claude responses

"Would a 20-year senior advisor be satisfied with this?"

Every Claude response is checked against quality thresholds BEFORE
it reaches the user. This module implements the self-reflection
pattern from Section 6 of the Master CoT:

  After generating any output, the system asks itself:
    Q1: "Would a 20-year senior advisor be satisfied with this?"
    Q2: "Does this output have data + context + signal + advice?"
    Q3: "Is this personalised to the user's situation?"
    Q4: "Is this honest about uncertainty?"
  If any answer is NO → rewrite before returning.

Quality thresholds (from config.py):
  situation  >= 100 chars  → What is happening must be substantial
  reasoning  >= 100 chars  → Why must explain causation
  action     >= 50 chars   → What to do must include price levels
  risk       >= 50 chars   → What could go wrong must be concrete
  lesson     >= 80 chars   → What to learn must be educational (full only)
  cot        >= 50 chars   → Reasoning trail must be present

Banned phrases:
  "N/A", "Not applicable", "Not available", "No data",
  "TBD", "To be determined", "None available"
  → Fields containing these are treated as EMPTY.

The checker returns an AnalysisQualityReport. If verdict is FAILED,
the caller raises AnalysisQualityError which triggers a retry with
the QUALITY_REMINDER prompt containing the specific issues found.

Verdict logic:
  PASSED   → All field checks pass + all self-reflection questions YES
  WARNING  → 1-2 minor issues (shallow fields) but core is solid
  FAILED   → Missing fields, banned content, or ≥3 issues → retry
"""

from __future__ import annotations

import logging
from datetime import datetime

from zoneinfo import ZoneInfo

from module2_analysis_engine.config import (
    QUALITY_BANNED_PHRASES,
    QUALITY_MIN_ACTION,
    QUALITY_MIN_COT,
    QUALITY_MIN_LESSON,
    QUALITY_MIN_REASONING,
    QUALITY_MIN_RISK,
    QUALITY_MIN_SITUATION,
)
from module2_analysis_engine.models import (
    AnalysisDepth,
    AnalysisQualityError,
    AnalysisQualityReport,
    MarketAnalysis,
    QualityVerdict,
    UserContext,
)

IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger("swing_advisor.quality_checker")


class QualityChecker:
    """Self-reflection quality gate for Claude's market analysis.

    This is the quality conscience of the advisor. It checks every
    response against thresholds and self-reflection questions before
    letting it reach the user. A senior advisor NEVER sends sloppy
    analysis — and neither does this bot.

    The checker runs two passes:
      Pass 1: Field-level checks (length, banned phrases, presence)
      Pass 2: Self-reflection questions (advisor quality, structure,
              personalisation, honesty)

    Usage:
        checker = QualityChecker()
        report = checker.check(analysis, user_context)
        if report.verdict == QualityVerdict.FAILED:
            raise AnalysisQualityError(
                missing_fields=report.missing_fields,
                shallow_fields=report.shallow_fields,
                retry_count=attempt,
            )
    """

    def check(
        self,
        analysis: MarketAnalysis,
        user_context: UserContext | None = None,
    ) -> AnalysisQualityReport:
        """Run the full quality gate on a MarketAnalysis.

        This is the main entry point. It performs:
          Pass 1: Field-level checks (length thresholds, banned phrases)
          Pass 2: Self-reflection questions (advisor quality)
          Final:  Determine verdict (PASSED / WARNING / FAILED)

        Args:
            analysis: The MarketAnalysis to check.
            user_context: Optional user context for personalisation check.

        Returns:
            AnalysisQualityReport with verdict and detailed issues.
        """
        issues: list[str] = []
        missing_fields: list[str] = []
        shallow_fields: list[str] = []
        is_full = analysis.analysis_depth == AnalysisDepth.FULL

        # ── Pass 1: Field-Level Checks ──
        situation_ok = self._check_field(
            field_name="situation",
            value=analysis.situation,
            min_length=QUALITY_MIN_SITUATION,
            issues=issues,
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
        )

        reasoning_ok = self._check_field(
            field_name="reasoning",
            value=analysis.reasoning,
            min_length=QUALITY_MIN_REASONING,
            issues=issues,
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
        )

        action_ok = self._check_field(
            field_name="action",
            value=analysis.action,
            min_length=QUALITY_MIN_ACTION,
            issues=issues,
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
        )

        risk_ok = self._check_field(
            field_name="risk",
            value=analysis.risk,
            min_length=QUALITY_MIN_RISK,
            issues=issues,
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
        )

        # Lesson is only required for full analysis depth
        if is_full:
            lesson_ok = self._check_field(
                field_name="lesson",
                value=analysis.lesson,
                min_length=QUALITY_MIN_LESSON,
                issues=issues,
                missing_fields=missing_fields,
                shallow_fields=shallow_fields,
            )
        else:
            lesson_ok = True

        # CoT reasoning check
        cot_present = self._check_field(
            field_name="cot_reasoning",
            value=analysis.cot_reasoning,
            min_length=QUALITY_MIN_COT,
            issues=issues,
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
        )

        # Banned phrases check across all text fields
        no_na_fields = self._check_banned_phrases(
            analysis=analysis,
            issues=issues,
        )

        # Personalisation check
        personalisation_present = self._check_personalisation(
            analysis=analysis,
            user_context=user_context,
            issues=issues,
        )

        # ── Pass 2: Self-Reflection Questions ──
        # Q1: Would a 20-year senior advisor be satisfied?
        advisor_satisfied = (
            situation_ok
            and reasoning_ok
            and action_ok
            and risk_ok
            and no_na_fields
        )
        if not advisor_satisfied:
            issues.append(
                "Q1 FAILED: A senior advisor would NOT be satisfied with this output. "
                "Missing depth in core fields."
            )

        # Q2: Does output have data + context + signal + advice?
        has_full_structure = (
            situation_ok    # data — what is happening
            and reasoning_ok  # context — why
            and action_ok     # signal + advice — what to do
            and risk_ok       # risk warning
        )
        if not has_full_structure:
            issues.append(
                "Q2 FAILED: Output does not have the complete structure "
                "(data + context + signal + advice + risk)."
            )

        # Q3: Is this personalised to the user?
        is_personalised = personalisation_present or user_context is None
        if not is_personalised:
            issues.append(
                "Q3 FAILED: Output is not personalised to the user's situation. "
                "Should reference user positions, capital, or risk tolerance."
            )

        # Q4: Is this honest about uncertainty?
        is_honest = self._check_honesty(analysis)
        if not is_honest:
            issues.append(
                "Q4 WARNING: Analysis shows extreme confidence without hedging language. "
                "A good advisor acknowledges uncertainty."
            )

        # ── Determine Verdict ──
        verdict = self._determine_verdict(
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
            issues=issues,
            advisor_satisfied=advisor_satisfied,
            has_full_structure=has_full_structure,
        )

        report = AnalysisQualityReport(
            verdict=verdict,
            situation_ok=situation_ok,
            reasoning_ok=reasoning_ok,
            action_ok=action_ok,
            risk_ok=risk_ok,
            lesson_ok=lesson_ok,
            cot_present=cot_present,
            no_na_fields=no_na_fields,
            personalisation_present=personalisation_present,
            advisor_satisfied=advisor_satisfied,
            has_full_structure=has_full_structure,
            is_personalised=is_personalised,
            is_honest=is_honest,
            missing_fields=missing_fields,
            shallow_fields=shallow_fields,
            issues=issues,
            checked_at=datetime.now(IST),
        )

        log_msg = (
            f"Quality check: {verdict.value}. "
            f"Missing: {missing_fields}. Shallow: {shallow_fields}. "
            f"Issues: {len(issues)}."
        )
        if verdict == QualityVerdict.FAILED:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return report

    def _check_field(
        self,
        field_name: str,
        value: str,
        min_length: int,
        issues: list[str],
        missing_fields: list[str],
        shallow_fields: list[str],
    ) -> bool:
        """Check a single field against its quality threshold.

        Three possible outcomes:
          → Field is empty/None → MISSING (critical failure)
          → Field is below min_length → SHALLOW (retryable)
          → Field meets threshold → OK

        Args:
            field_name: Name of the field being checked.
            value: Field value (string).
            min_length: Minimum character count.
            issues: Issues list (appended to).
            missing_fields: Missing fields list (appended to).
            shallow_fields: Shallow fields list (appended to).

        Returns:
            True if field passes the check.
        """
        # Treat None or empty as missing
        if not value or not value.strip():
            missing_fields.append(field_name)
            issues.append(
                f"MISSING: '{field_name}' is empty. "
                f"A senior advisor always provides {field_name}."
            )
            return False

        cleaned = value.strip()

        # Check for banned phrases — treat as effectively empty
        for phrase in QUALITY_BANNED_PHRASES:
            if cleaned.lower() == phrase.lower():
                missing_fields.append(field_name)
                issues.append(
                    f"BANNED: '{field_name}' contains only '{phrase}'. "
                    f"This is not real content — field is effectively empty."
                )
                return False

        # Check length threshold
        if len(cleaned) < min_length:
            shallow_fields.append(field_name)
            issues.append(
                f"SHALLOW: '{field_name}' is {len(cleaned)} chars "
                f"(minimum {min_length}). Needs more depth and detail."
            )
            return False

        return True

    def _check_banned_phrases(
        self,
        analysis: MarketAnalysis,
        issues: list[str],
    ) -> bool:
        """Check all text fields for banned filler phrases.

        Banned phrases ("N/A", "Not applicable", etc.) indicate
        Claude is filling in fields without real content. Any field
        containing a banned phrase as a substring is flagged.

        Unlike _check_field which checks if the ENTIRE field equals
        a banned phrase, this checks for substrings within longer text.

        Args:
            analysis: MarketAnalysis to check.
            issues: Issues list (appended to).

        Returns:
            True if no banned phrases found in any field.
        """
        fields_to_check = {
            "situation": analysis.situation,
            "reasoning": analysis.reasoning,
            "action": analysis.action,
            "risk": analysis.risk,
            "lesson": analysis.lesson,
            "user_impact": analysis.user_impact,
            "cot_reasoning": analysis.cot_reasoning,
        }

        found_banned = False
        for field_name, value in fields_to_check.items():
            if not value:
                continue
            value_lower = value.lower()
            for phrase in QUALITY_BANNED_PHRASES:
                if phrase.lower() in value_lower:
                    issues.append(
                        f"BANNED PHRASE: '{field_name}' contains '{phrase}'. "
                        f"Replace with real analysis content."
                    )
                    found_banned = True

        return not found_banned

    def _check_personalisation(
        self,
        analysis: MarketAnalysis,
        user_context: UserContext | None,
        issues: list[str],
    ) -> bool:
        """Check if the analysis is personalised to the user.

        A good senior advisor always relates the market to the
        user's specific situation. This check looks for evidence
        of personalisation in the user_impact and action fields.

        Personalisation indicators:
          → User's display_name mentioned
          → Reference to open positions (ticker symbols)
          → Reference to capital or risk tolerance
          → Specific advice tied to user's holdings

        Args:
            analysis: MarketAnalysis to check.
            user_context: User context (None if no user data available).
            issues: Issues list (appended to).

        Returns:
            True if personalisation is detected.
        """
        if user_context is None:
            # No user context → cannot personalise, not a failure
            return True

        # Build a set of personalisation markers
        markers: list[str] = []

        if user_context.display_name:
            markers.append(user_context.display_name.lower())

        # Check for position ticker references
        if user_context.open_positions:
            for position in user_context.open_positions:
                if hasattr(position, "ticker"):
                    markers.append(position.ticker.lower())
                elif isinstance(position, str):
                    markers.append(position.lower())

        # Check for risk tolerance reference
        if user_context.risk_tolerance:
            markers.append(user_context.risk_tolerance.lower())

        if not markers:
            return True

        # Search user_impact and action for any marker
        search_text = (
            f"{analysis.user_impact} {analysis.action} {analysis.lesson}"
        ).lower()

        for marker in markers:
            if marker in search_text:
                return True

        return False

    def _check_honesty(self, analysis: MarketAnalysis) -> bool:
        """Check if the analysis is honest about uncertainty.

        A senior advisor never speaks in absolutes. They use
        hedging language to acknowledge uncertainty. This check
        flags analyses that sound overconfident.

        Red flags:
          → "will definitely" → should be "could" or "may"
          → "guaranteed" → nothing is guaranteed in markets
          → Very high confidence (>0.95) with strong mood calls

        Honesty indicators (at least one should be present):
          → "could", "may", "might", "likely", "possibly"
          → "uncertain", "risk", "caution", "monitor"

        Args:
            analysis: MarketAnalysis to check.

        Returns:
            True if analysis shows honest uncertainty acknowledgment.
        """
        # Overconfidence red flags
        overconfident_phrases = [
            "will definitely",
            "guaranteed",
            "certain to",
            "impossible to lose",
            "sure to rise",
            "sure to fall",
            "100% chance",
            "can't go wrong",
            "no risk",
            "zero risk",
        ]

        search_text = (
            f"{analysis.situation} {analysis.reasoning} "
            f"{analysis.action} {analysis.risk}"
        ).lower()

        for phrase in overconfident_phrases:
            if phrase in search_text:
                return False

        # For extreme confidence on strong mood calls, require hedging
        if analysis.mood_confidence > 0.95:
            hedging_words = [
                "could", "may", "might", "likely", "possibly",
                "uncertain", "caution", "monitor", "watch",
                "risk", "however", "although", "but",
            ]
            has_hedging = any(word in search_text for word in hedging_words)
            if not has_hedging:
                return False

        return True

    def _determine_verdict(
        self,
        missing_fields: list[str],
        shallow_fields: list[str],
        issues: list[str],
        advisor_satisfied: bool,
        has_full_structure: bool,
    ) -> QualityVerdict:
        """Determine the overall quality verdict.

        Verdict logic:
          FAILED  → Any missing fields (critical data gap)
                  → Core structure incomplete (Q2 failed)
                  → 3 or more issues total
          WARNING → 1-2 shallow fields, but core structure OK
                  → Advisor not fully satisfied but usable
          PASSED  → All checks pass

        Args:
            missing_fields: Fields that are completely missing.
            shallow_fields: Fields below quality threshold.
            issues: All issues found.
            advisor_satisfied: Q1 result.
            has_full_structure: Q2 result.

        Returns:
            QualityVerdict.PASSED, WARNING, or FAILED.
        """
        # Critical failures → FAILED
        if missing_fields:
            return QualityVerdict.FAILED

        if not has_full_structure:
            return QualityVerdict.FAILED

        if len(issues) >= 3:
            return QualityVerdict.FAILED

        # Minor issues → WARNING
        if shallow_fields or not advisor_satisfied:
            return QualityVerdict.WARNING

        if issues:
            return QualityVerdict.WARNING

        # All clear → PASSED
        return QualityVerdict.PASSED

    def format_issues_for_retry(
        self,
        report: AnalysisQualityReport,
    ) -> str:
        """Format quality issues into a string for the retry prompt.

        This output is inserted into the QUALITY_REMINDER prompt
        template to tell Claude exactly what to fix. The retry
        prompt uses {quality_issues} placeholder.

        Args:
            report: The quality report with issues.

        Returns:
            Formatted string describing all issues for Claude.
        """
        parts: list[str] = []

        if report.missing_fields:
            parts.append(
                f"MISSING FIELDS (must add): {', '.join(report.missing_fields)}."
            )

        if report.shallow_fields:
            parts.append(
                f"SHALLOW FIELDS (need more depth): {', '.join(report.shallow_fields)}."
            )

        # Add specific issue descriptions (skip Q1-Q4 meta-issues)
        field_issues = [
            issue for issue in report.issues
            if not issue.startswith("Q1") and not issue.startswith("Q2")
            and not issue.startswith("Q3") and not issue.startswith("Q4")
        ]
        if field_issues:
            parts.append("SPECIFIC ISSUES:")
            for issue in field_issues:
                parts.append(f"  → {issue}")

        if not parts:
            parts.append("General quality below advisor standard. Add more depth.")

        return "\n".join(parts)


# Module-level singleton — used across the analysis engine
quality_checker = QualityChecker()
