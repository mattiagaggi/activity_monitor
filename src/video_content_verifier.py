#!/usr/bin/env python3
"""
Video Content Verifier

Extracts frames from video at specific timestamps and uses OCR to verify
that the visual content matches the activity and keystroke logs.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import tempfile


class VideoContentVerifier:
    """Verifies video content matches activity logs using OCR."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_verify_"))

    def extract_frame(self, video_path: Path, timestamp_seconds: float,
                     output_path: Path) -> bool:
        """
        Extract a single frame from video at specified timestamp.

        Args:
            video_path: Path to video file
            timestamp_seconds: Time in seconds from start of video
            output_path: Where to save the frame

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use ffmpeg to extract frame
            result = subprocess.run([
                'ffmpeg',
                '-ss', str(timestamp_seconds),
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',
                '-y',
                str(output_path)
            ], capture_output=True, text=True, timeout=10)

            return result.returncode == 0 and output_path.exists()
        except Exception as e:
            print(f"Error extracting frame: {e}")
            return False

    def ocr_frame(self, image_path: Path) -> str:
        """
        Perform OCR on an image using tesseract.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        try:
            result = subprocess.run([
                'tesseract',
                str(image_path),
                'stdout'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return result.stdout
            return ""
        except Exception as e:
            print(f"Error performing OCR: {e}")
            return ""

    def calculate_video_timestamp(self, period_start: datetime,
                                  activity_timestamp: datetime,
                                  cumulative_seconds_before_period: int) -> float:
        """
        Calculate the timestamp in the video for a given activity timestamp.

        Args:
            period_start: When this period starts in real time
            activity_timestamp: When the activity occurred in real time
            cumulative_seconds_before_period: Total active seconds in previous periods

        Returns:
            Timestamp in video (seconds from start)
        """
        seconds_into_period = (activity_timestamp - period_start).total_seconds()
        return cumulative_seconds_before_period + seconds_into_period

    def verify_video_content(self, video_path: Path, match_json_path: Path,
                           sample_count: int = 10) -> Dict:
        """
        Verify video content by sampling frames and comparing with activity logs.

        Args:
            video_path: Path to video file
            match_json_path: Path to video match JSON file
            sample_count: Number of frames to sample per period

        Returns:
            Dictionary with verification results
        """
        # Load match data
        with open(match_json_path) as f:
            match_data = json.load(f)

        results = {
            'video_path': str(video_path),
            'total_periods': len(match_data['periods']),
            'samples': [],
            'verification_summary': {}
        }

        cumulative_seconds = 0

        for period_idx, period in enumerate(match_data['periods']):
            period_start = datetime.fromisoformat(period['start'])
            period_end = datetime.fromisoformat(period['end'])
            period_duration = period['active_time_seconds']

            print(f"\n{'='*80}")
            print(f"Period {period_idx + 1}: {period_start.strftime('%H:%M:%S')} - {period_end.strftime('%H:%M:%S')}")
            print(f"{'='*80}")

            # Sample activities from this period
            activities = period['activities']

            # Sample evenly across the period
            sample_interval = max(1, len(activities) // sample_count)
            sampled_activities = activities[::sample_interval][:sample_count]

            for act_idx, activity in enumerate(sampled_activities):
                activity_time = datetime.fromisoformat(activity['timestamp'])

                # Calculate video timestamp
                video_ts = self.calculate_video_timestamp(
                    period_start, activity_time, cumulative_seconds
                )

                print(f"\nSample {act_idx + 1}/{len(sampled_activities)}:")
                print(f"  Real time: {activity_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Video timestamp: {video_ts:.1f}s ({video_ts/60:.1f}m)")
                print(f"  App: {activity['app']}")
                print(f"  Title: {activity['title'][:50]}")

                # Extract frame
                frame_path = self.temp_dir / f"period{period_idx}_sample{act_idx}.png"

                if self.extract_frame(video_path, video_ts, frame_path):
                    print(f"  ✓ Frame extracted: {frame_path}")

                    # Perform OCR
                    ocr_text = self.ocr_frame(frame_path)

                    # Store result
                    sample_result = {
                        'period': period_idx + 1,
                        'real_timestamp': activity['timestamp'],
                        'video_timestamp': video_ts,
                        'expected_app': activity['app'],
                        'expected_title': activity['title'],
                        'frame_path': str(frame_path),
                        'ocr_text': ocr_text[:500]  # Limit length
                    }

                    results['samples'].append(sample_result)

                    # Check if OCR text contains app-related keywords
                    app_name = activity['app'].split('.')[-1].lower()
                    title_words = activity['title'].lower().split()

                    ocr_lower = ocr_text.lower()

                    app_match = app_name in ocr_lower
                    title_match = any(word in ocr_lower for word in title_words if len(word) > 3)

                    print(f"  OCR text preview: {ocr_text[:100].strip()}")
                    print(f"  App match: {'✓' if app_match else '✗'} ({app_name})")
                    print(f"  Title match: {'✓' if title_match else '✗'}")

                    sample_result['app_match'] = app_match
                    sample_result['title_match'] = title_match
                else:
                    print(f"  ✗ Failed to extract frame")

            # Add this period's duration to cumulative
            cumulative_seconds += period_duration

        # Calculate verification summary
        total_samples = len(results['samples'])
        app_matches = sum(1 for s in results['samples'] if s.get('app_match', False))
        title_matches = sum(1 for s in results['samples'] if s.get('title_match', False))

        results['verification_summary'] = {
            'total_samples': total_samples,
            'app_matches': app_matches,
            'title_matches': title_matches,
            'app_match_rate': app_matches / total_samples if total_samples > 0 else 0,
            'title_match_rate': title_matches / total_samples if total_samples > 0 else 0
        }

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate human-readable verification report."""
        lines = []
        lines.append("=" * 80)
        lines.append("VIDEO CONTENT VERIFICATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Video: {results['video_path']}")
        lines.append(f"Periods analyzed: {results['total_periods']}")
        lines.append(f"Frames sampled: {results['verification_summary']['total_samples']}")
        lines.append("")

        summary = results['verification_summary']
        lines.append("VERIFICATION RESULTS:")
        lines.append(f"  App matches: {summary['app_matches']}/{summary['total_samples']} ({summary['app_match_rate']*100:.1f}%)")
        lines.append(f"  Title matches: {summary['title_matches']}/{summary['total_samples']} ({summary['title_match_rate']*100:.1f}%)")
        lines.append("")

        if summary['app_match_rate'] > 0.7:
            lines.append("✅ VERIFICATION PASSED: Video content matches activity logs")
        elif summary['app_match_rate'] > 0.4:
            lines.append("⚠️  PARTIAL MATCH: Some discrepancies found")
        else:
            lines.append("❌ VERIFICATION FAILED: Significant discrepancies")

        lines.append("")
        lines.append("SAMPLE DETAILS:")
        lines.append("-" * 80)

        for i, sample in enumerate(results['samples'][:20], 1):  # Show first 20
            lines.append(f"\nSample {i}:")
            lines.append(f"  Time: {sample['real_timestamp']} (video: {sample['video_timestamp']:.1f}s)")
            lines.append(f"  Expected: {sample['expected_app'].split('.')[-1]} - {sample['expected_title'][:40]}")
            lines.append(f"  Match: App={'✓' if sample.get('app_match') else '✗'} Title={'✓' if sample.get('title_match') else '✗'}")
            if sample.get('ocr_text'):
                preview = sample['ocr_text'][:80].replace('\n', ' ').strip()
                lines.append(f"  OCR: {preview}...")

        if len(results['samples']) > 20:
            lines.append(f"\n... ({len(results['samples']) - 20} more samples)")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def cleanup(self):
        """Remove temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def main():
    """Example usage."""
    verifier = VideoContentVerifier()

    base_dir = Path(__file__).parent.parent / "data"

    sessions = [
        {
            'name': 'Session 1 (sassion1)',
            'video': base_dir / 'sassion1' / 'video.mp4',
            'match': base_dir / 'sassion1' / 'video_match.json'
        },
        {
            'name': 'Session 2',
            'video': base_dir / 'session2' / 'video.mp4',
            'match': base_dir / 'session2' / 'video_match.json'
        }
    ]

    try:
        for session in sessions:
            print(f"\n{'='*80}")
            print(f"Verifying: {session['name']}")
            print(f"{'='*80}")

            if not session['video'].exists():
                print(f"❌ Video not found: {session['video']}")
                continue

            if not session['match'].exists():
                print(f"❌ Match data not found: {session['match']}")
                continue

            # Verify content
            results = verifier.verify_video_content(
                session['video'],
                session['match'],
                sample_count=5  # Sample 5 frames per period
            )

            # Generate report
            print("\n" + verifier.generate_report(results))

            # Save results
            output_path = session['video'].parent / f"{session['video'].stem}_verification.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n✅ Verification data saved to: {output_path}")

    finally:
        verifier.cleanup()


if __name__ == "__main__":
    main()
