"""
Test Harness: Verify Your Grounded Report
===========================================
Loads your X-ray image and grounded report JSON, runs the verifier,
and generates a comprehensive verification report with visualizations.
"""

import json
from pathlib import Path
from typing import Dict
from PIL import Image, ImageDraw

# Import the verifier pieces (adjust path as needed)
from finding_verifier import FindingVerifier, VerificationStatus
from Torchxrayvision_integration import TorchXRayVisionVerifier


class ReportVerificationHarness:
    """End-to-end harness for testing the verifier on real data"""

    def __init__(self, image_path: str, report_path: str, device: str = 'cpu'):
        """
        Initialize the test harness.

        Args:
            image_path: Path to annotated X-ray image
            report_path: Path to grounded report JSON
            device: kept for API compatibility. TorchXRayVisionVerifier currently
                    picks cuda/cpu automatically (torch.cuda.is_available()), so
                    this is not yet wired through — flagging in case you want to
                    force CPU even when a GPU is present.
        """
        self.image_path = Path(image_path)
        self.report_path = Path(report_path)
        self.device = device

        # Load the report
        print(f"[Harness] Loading report from {self.report_path}")
        with open(self.report_path, 'r') as f:
            self.report = json.load(f)

        print(f"[Harness] Report loaded: {len(self.report.get('findings', []))} findings found")

        # Build the image-grounded pathology model, then wrap it in the
        # higher-level FindingVerifier that applies the guardrail logic.
        txv_verifier = TorchXRayVisionVerifier()
        self.verifier = FindingVerifier(txv_verifier)

    def run_verification(self) -> Dict:
        """Run the full verification pipeline"""
        findings = self.report.get('findings', [])
        impression = self.report.get('impression', '')

        print(f"\n{'='*70}")
        print(f"VERIFICATION PIPELINE")
        print(f"{'='*70}\n")

        # Step 1: Verify each finding
        print(f"Step 1: Verifying findings against image...")
        results = self.verifier.verify_findings(str(self.image_path), findings)

        # Step 2: Build a summary report from the VerificationResult objects
        print(f"\nStep 2: Generating verification report...")
        verification_report = self._build_report(results, impression)

        print(f"\nStep 3: Summarizing results...")
        self._print_summary(verification_report)

        return {
            'verification_report': verification_report,
            'results': results,
            'original_report': self.report,
        }

    def _build_report(self, results, impression: str) -> Dict:
        """Turn a list of VerificationResult objects into a summary report dict"""
        counts = {status: 0 for status in VerificationStatus}
        for r in results:
            counts[r.status] += 1

        total = len(results)
        unsupported = counts[VerificationStatus.UNSUPPORTED]

        return {
            'impression': impression,
            'summary': {
                'total_findings': total,
                'supported_findings': counts[VerificationStatus.SUPPORTED],
                'kept_findings': counts[VerificationStatus.SUPPORTED],
                'flagged_findings': counts[VerificationStatus.UNCERTAIN],
                'suppressed_findings': unsupported,
                'abstained_findings': counts[VerificationStatus.ABSTAIN],
                'hallucination_rate': unsupported / max(1, total),
            },
            'findings': [r.to_dict() for r in results],
        }

    def _print_summary(self, report: Dict):
        """Pretty-print verification summary"""
        summary = report.get('summary', {})

        print(f"\n{'-'*70}")
        print(f"VERIFICATION SUMMARY")
        print(f"{'-'*70}")
        print(f"Total findings:       {summary.get('total_findings', 0)}")
        print(f"Supported:            {summary.get('supported_findings', 0)}")
        print(f"Kept (safe):          {summary.get('kept_findings', 0)} \u2713")
        print(f"Flagged (uncertain):  {summary.get('flagged_findings', 0)} \u26a0")
        print(f"Suppressed (halluci.): {summary.get('suppressed_findings', 0)} \u2717")
        print(f"Abstained:            {summary.get('abstained_findings', 0)} ?")
        print(f"Hallucination rate:   {summary.get('hallucination_rate', 0):.1%}")
        print(f"{'-'*70}\n")

    def save_verification_report(self, output_path: str):
        """Run verification and save the report"""
        results = self.run_verification()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save summary report
        with open(output_file, 'w') as f:
            json.dump(results['verification_report'], f, indent=2)

        print(f"\n[Harness] Report saved to {output_file}")

        # Save per-finding results
        findings_file = output_file.parent / f"{output_file.stem}_detailed.json"
        with open(findings_file, 'w') as f:
            json.dump({
                'findings': [r.to_dict() for r in results['results']]
            }, f, indent=2)

        print(f"[Harness] Detailed findings saved to {findings_file}")

        return results

    def visualize_verification(self, output_path: str):
        """
        Create a visualization showing which findings were verified and how.
        Generates an annotated image with verification status overlays.
        """
        results = self.run_verification()

        # Load original image
        img = Image.open(self.image_path)
        img_with_verification = img.copy().convert('RGBA')
        draw = ImageDraw.Draw(img_with_verification, 'RGBA')

        w, h = img.size
        findings = self.report.get('findings', [])

        # Color coding by verification status
        status_colors = {
            VerificationStatus.SUPPORTED: (0, 255, 0, 150),     # Green
            VerificationStatus.UNCERTAIN: (255, 165, 0, 150),   # Orange
            VerificationStatus.UNSUPPORTED: (255, 0, 0, 150),   # Red
            VerificationStatus.ABSTAIN: (128, 128, 255, 150),   # Blue
        }

        print(f"\n[Harness] Creating verification visualization...")

        for finding, result in zip(findings, results['results']):
            bbox = finding.get('bounding_box', {})
            if not bbox:
                continue

            # Convert normalized coords to pixel coords
            x1 = int(bbox['x'] * w)
            y1 = int(bbox['y'] * h)
            x2 = int((bbox['x'] + bbox['width']) * w)
            y2 = int((bbox['y'] + bbox['height']) * h)

            # Draw rectangle with status color
            color = status_colors.get(result.status, (255, 255, 255, 150))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Add label with status
            label = f"{finding['finding_name'][:15]}\n{result.status.value.upper()}"
            draw.text((x1 + 5, y1 + 5), label, fill=color)

        # Save visualization
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        img_with_verification.save(output_file)

        print(f"[Harness] Visualization saved to {output_file}")

        return img_with_verification


def main():
    """Main entry point for the test harness"""

    # Paths to your data
    image_path = "output/annotated.png"
    report_path = "output/reports_grounded.json"

    # Output paths
    output_report = "output/verification_report.json"
    output_visualization = "output/verification_visualization.png"

    print("\n" + "="*70)
    print("GROUNDED REPORT VERIFICATION TEST HARNESS")
    print("="*70 + "\n")

    # Create harness
    harness = ReportVerificationHarness(
        image_path=image_path,
        report_path=report_path,
        device='cpu'
    )

    # Run verification and save report
    print("\n[Main] Running verification pipeline...")
    harness.save_verification_report(output_report)

    # Create visualization
    print("\n[Main] Creating visualization...")
    harness.visualize_verification(output_visualization)

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  \u2022 Verification report: {output_report}")
    print(f"  \u2022 Detailed findings:   {output_report.replace('.json', '_detailed.json')}")
    print(f"  \u2022 Visualization:       {output_visualization}")

    return 0


if __name__ == '__main__':
    main()
