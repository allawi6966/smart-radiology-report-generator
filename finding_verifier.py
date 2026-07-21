"""
Finding Verifier Module
Orchestrates image-grounded verification of radiological findings.
Implements hallucination detection and guardrail logic.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class VerificationStatus(Enum):
    """Status of finding verification."""
    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"
    ABSTAIN = "abstain"


@dataclass
class VerificationResult:
    """Result of verifying a single finding."""
    finding_id: str
    finding_name: str
    location: str
    status: VerificationStatus
    evidence_score: float
    model_confidence: float
    verifier_confidence: float
    explanation: str
    matched_pathologies: List[Tuple[str, float]]
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'finding_id': self.finding_id,
            'finding_name': self.finding_name,
            'location': self.location,
            'status': self.status.value,
            'evidence_score': float(self.evidence_score),
            'model_confidence': float(self.model_confidence),
            'verifier_confidence': float(self.verifier_confidence),
            'explanation': self.explanation,
            'matched_pathologies': [(p, float(s)) for p, s in self.matched_pathologies],
            'recommendation': self.recommendation
        }


class FindingVerifier:
    """
    Main verification pipeline for radiological findings.
    Combines model confidence with image-grounded evidence.
    """
    
    def __init__(self, 
                 torchxray_verifier,
                 evidence_threshold: float = 0.40,
                 uncertainty_threshold: float = 0.25,
                 abstention_enabled: bool = True):
        """
        Initialize the finding verifier.
        
        Args:
            torchxray_verifier: TorchXRayVisionVerifier instance
            evidence_threshold: Score above which a finding is considered supported (0-1)
            uncertainty_threshold: Score between uncertainty_threshold and evidence_threshold
                                   results in UNCERTAIN status
            abstention_enabled: If True, abstain on low confidence instead of falsely supporting
        """
        self.verifier = torchxray_verifier
        self.evidence_threshold = evidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.abstention_enabled = abstention_enabled
        
    def verify_findings(self, 
                       image_path: str,
                       findings: List[Dict[str, Any]]) -> List[VerificationResult]:
        """
        Verify multiple findings from a structured report.
        
        Args:
            image_path: Path to chest X-ray image
            findings: List of finding dicts from structured report
                     Each dict should have: id, finding_name, location, bounding_box, 
                                          description, severity, confidence
            
        Returns:
            List of VerificationResult objects
        """
        results = []
        
        for finding in findings:
            result = self.verify_single_finding(
                image_path=image_path,
                finding=finding
            )
            results.append(result)
        
        return results
    
    def verify_single_finding(self, 
                             image_path: str,
                             finding: Dict[str, Any]) -> VerificationResult:
        """
        Verify a single finding against the image.
        
        Args:
            image_path: Path to chest X-ray
            finding: Finding dict with structure from structured report
            
        Returns:
            VerificationResult with detailed verification outcome
        """
        finding_id = finding.get('id', 'unknown')
        finding_name = finding.get('finding_name', 'Unknown Finding')
        location = finding.get('location', 'Unknown')
        description = finding.get('description', '')
        model_confidence = finding.get('confidence', 0.5)
        bbox = finding.get('bounding_box', None)
        
        # Get image-grounded evidence
        verification = self.verifier.verify_finding(
            image_path_or_array=image_path,
            finding_name=finding_name,
            finding_description=description,
            bbox=bbox,
            use_region=True
        )
        
        evidence_score = verification['evidence_score']
        top_pathologies = verification['top_pathologies'][:3]  # Top 3
        
        # Determine status based on evidence + model confidence
        status, explanation = self._determine_status(
            evidence_score=evidence_score,
            model_confidence=model_confidence,
            finding_name=finding_name
        )
        
        # Calculate verifier confidence (how confident are we in this verification?)
        verifier_confidence = self._calculate_verifier_confidence(
            evidence_score=evidence_score,
            model_confidence=model_confidence,
            region_available=bbox is not None
        )
        
        # Generate recommendation for guardrail
        recommendation = self._generate_recommendation(
            status=status,
            evidence_score=evidence_score,
            model_confidence=model_confidence,
            verifier_confidence=verifier_confidence
        )
        
        return VerificationResult(
            finding_id=finding_id,
            finding_name=finding_name,
            location=location,
            status=status,
            evidence_score=evidence_score,
            model_confidence=model_confidence,
            verifier_confidence=verifier_confidence,
            explanation=explanation,
            matched_pathologies=top_pathologies,
            recommendation=recommendation
        )
    
    def _determine_status(self, 
                         evidence_score: float,
                         model_confidence: float,
                         finding_name: str) -> Tuple[VerificationStatus, str]:
        """Determine verification status based on evidence and model confidence."""
        
        # Strong evidence: finding is clearly supported
        if evidence_score >= self.evidence_threshold:
            return (
                VerificationStatus.SUPPORTED,
                f"Image evidence strongly supports {finding_name} (score: {evidence_score:.2f})"
            )
        
        # Weak evidence but high model confidence: uncertain
        elif evidence_score >= self.uncertainty_threshold:
            if model_confidence >= 0.7:
                return (
                    VerificationStatus.UNCERTAIN,
                    f"Image shows weak evidence; model is confident. Needs radiologist review."
                )
            else:
                return (
                    VerificationStatus.UNSUPPORTED,
                    f"Insufficient image evidence and low model confidence"
                )
        
        # No clear evidence
        else:
            if self.abstention_enabled and model_confidence < 0.6:
                return (
                    VerificationStatus.ABSTAIN,
                    f"Cannot verify from image; low model confidence. Abstaining."
                )
            else:
                return (
                    VerificationStatus.UNSUPPORTED,
                    f"No image evidence for {finding_name}"
                )
    
    def _calculate_verifier_confidence(self,
                                      evidence_score: float,
                                      model_confidence: float,
                                      region_available: bool) -> float:
        """
        Calculate how confident we are in the verification result.
        High verifier_confidence means the verification decision is reliable.
        """
        # Base confidence from evidence agreement with model
        agreement = 1.0 - abs(evidence_score - model_confidence)
        
        # Boost if region was available for targeted analysis
        region_boost = 0.1 if region_available else 0.0
        
        # Combine
        confidence = min(0.95, (agreement + region_boost) / 1.1)
        
        return confidence
    
    def _generate_recommendation(self,
                                status: VerificationStatus,
                                evidence_score: float,
                                model_confidence: float,
                                verifier_confidence: float) -> str:
        """Generate actionable recommendation for guardrail."""
        
        if status == VerificationStatus.SUPPORTED:
            return "✓ INCLUDE: Finding is well-grounded in image"
        
        elif status == VerificationStatus.UNCERTAIN:
            return f"⚠ REVIEW: Weak image evidence ({evidence_score:.2f}), high model confidence ({model_confidence:.2f}). Recommend radiologist review before inclusion."
        
        elif status == VerificationStatus.UNSUPPORTED:
            if evidence_score < 0.1:
                return "✗ SUPPRESS: Finding lacks image evidence (potential hallucination). Recommend removal."
            else:
                return "✗ FLAG: Insufficient evidence. Recommend exclusion or verification."
        
        elif status == VerificationStatus.ABSTAIN:
            return "⊥ ABSTAIN: Cannot reliably verify. Recommend exclusion to maintain safety."
        
        return "? UNKNOWN: Unable to generate recommendation"


class GuardrailPipeline:
    """
    End-to-end guardrail pipeline that filters/flags findings based on verification.
    """
    
    def __init__(self, verifier: FindingVerifier):
        self.verifier = verifier
    
    def process_report(self, 
                      image_path: str,
                      structured_report: Dict[str, Any],
                      suppress_unsupported: bool = False,
                      suppress_uncertain: bool = False) -> Dict[str, Any]:
        """
        Process a structured report through the guardrail.
        
        Args:
            image_path: Path to chest X-ray
            structured_report: Report dict with 'findings' and 'impression'
            suppress_unsupported: If True, remove unsupported findings from output
            suppress_uncertain: If True, remove uncertain findings from output
            
        Returns:
            Guardrailed report with verification results and recommendations
        """
        findings = structured_report.get('findings', [])
        impression = structured_report.get('impression', '')
        
        # Verify all findings
        verification_results = self.verifier.verify_findings(image_path, findings)
        
        # Categorize findings
        supported = []
        uncertain = []
        unsupported = []
        abstained = []
        
        for result in verification_results:
            if result.status == VerificationStatus.SUPPORTED:
                supported.append(result)
            elif result.status == VerificationStatus.UNCERTAIN:
                uncertain.append(result)
            elif result.status == VerificationStatus.UNSUPPORTED:
                unsupported.append(result)
            elif result.status == VerificationStatus.ABSTAIN:
                abstained.append(result)
        
        # Build guardrailed report
        guardrailed_findings = []
        
        if not suppress_unsupported:
            guardrailed_findings.extend([asdict(r) for r in supported])
        else:
            guardrailed_findings.extend([asdict(r) for r in supported])
        
        if not suppress_uncertain:
            guardrailed_findings.extend([asdict(r) for r in uncertain])
        
        if not suppress_unsupported:
            guardrailed_findings.extend([asdict(r) for r in unsupported])
        
        if not suppress_unsupported:
            guardrailed_findings.extend([asdict(r) for r in abstained])
        
        return {
            'original_impression': impression,
            'verification_summary': {
                'total_findings': len(findings),
                'supported': len(supported),
                'uncertain': len(uncertain),
                'unsupported': len(unsupported),
                'abstained': len(abstained),
                'hallucination_detected': len(unsupported) > 0,
                'hallucination_rate': len(unsupported) / max(1, len(findings))
            },
            'findings': guardrailed_findings,
            'guardrail_notes': self._generate_notes(supported, uncertain, unsupported, abstained)
        }
    
    def _generate_notes(self, 
                       supported: List[VerificationResult],
                       uncertain: List[VerificationResult],
                       unsupported: List[VerificationResult],
                       abstained: List[VerificationResult]) -> str:
        """Generate narrative notes about verification."""
        notes = []
        
        if unsupported:
            findings_list = ", ".join([f.finding_name for f in unsupported])
            notes.append(f"⚠ HALLUCINATIONS DETECTED: {findings_list} lack image evidence")
        
        if uncertain:
            notes.append(f"⚠ UNCERTAIN FINDINGS: {len(uncertain)} findings require radiologist review")
        
        if abstained:
            notes.append(f"⊥ ABSTAINED: {len(abstained)} findings could not be reliably verified")
        
        if supported:
            notes.append(f"✓ VERIFIED: {len(supported)} findings are well-grounded in image")
        
        return " | ".join(notes) if notes else "No verification issues detected."


if __name__ == "__main__":
    print("Finding Verifier Module")
    print("Use this module in test_harness.py")