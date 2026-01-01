#!/usr/bin/env python3
"""
Video-Activity Matching Module

Matches video recordings with activity and keystroke data to identify
which time periods are captured in each video.
"""

import csv
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ActivityRecord:
    """Represents a single activity entry."""
    timestamp: datetime
    duration: int  # seconds
    app: str
    title: str
    employee: str


@dataclass
class KeystrokeRecord:
    """Represents a single keystroke entry."""
    timestamp: datetime
    app: str
    keystroke: str
    employee: str


@dataclass
class ActivePeriod:
    """Represents a continuous period of activity."""
    start: datetime
    end: datetime
    active_time: int  # Total active seconds (sum of activity durations)
    activities: List[ActivityRecord] = field(default_factory=list)
    keystrokes: List[KeystrokeRecord] = field(default_factory=list)

    @property
    def timespan(self) -> int:
        """Total timespan in seconds from start to end."""
        return int((self.end - self.start).total_seconds())

    def get_top_apps(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get top N apps by duration."""
        app_durations = {}
        for activity in self.activities:
            app_durations[activity.app] = app_durations.get(activity.app, 0) + activity.duration

        return sorted(app_durations.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_keystroke_count(self) -> int:
        """Get total number of keystroke records in this period."""
        return len(self.keystrokes)


@dataclass
class VideoMatch:
    """Represents a match between video and activity periods."""
    video_path: Path
    video_duration: int  # seconds
    matched_periods: List[ActivePeriod]
    match_accuracy: float  # 0-1, how well the periods match the video duration

    @property
    def total_matched_time(self) -> int:
        """Total active time across all matched periods."""
        return sum(p.active_time for p in self.matched_periods)

    @property
    def time_difference(self) -> int:
        """Difference between video duration and matched time in seconds."""
        return abs(self.video_duration - self.total_matched_time)


class VideoActivityMatcher:
    """Matches video recordings with activity and keystroke data."""

    def __init__(self, max_gap_seconds: int = 300):
        """
        Initialize the matcher.

        Args:
            max_gap_seconds: Maximum gap between activities to consider them
                           part of the same continuous period (default: 5 minutes)
        """
        self.max_gap_seconds = max_gap_seconds

    def get_video_duration(self, video_path: Path) -> Optional[int]:
        """
        Get video duration in seconds using mdls (macOS) or fallback methods.

        Args:
            video_path: Path to the video file

        Returns:
            Duration in seconds, or None if unable to determine
        """
        try:
            # Try macOS mdls first
            result = subprocess.run(
                ['mdls', '-name', 'kMDItemDurationSeconds', str(video_path)],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0 and '=' in result.stdout:
                duration_str = result.stdout.strip().split('=')[1].strip()
                return int(float(duration_str))
        except Exception:
            pass

        # Try ffprobe as fallback
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                return int(float(result.stdout.strip()))
        except Exception:
            pass

        return None

    def load_activities(self, csv_path: Path) -> List[ActivityRecord]:
        """
        Load activity records from CSV file.

        Args:
            csv_path: Path to ActivityGrid CSV file

        Returns:
            List of ActivityRecord objects
        """
        activities = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.strptime(row['Timestamp'], '%Y-%m-%d %H:%M:%S')

                    # Parse duration HH:MM:SS
                    time_parts = row['Time'].split(':')
                    duration = (int(time_parts[0]) * 3600 +
                               int(time_parts[1]) * 60 +
                               int(time_parts[2]))

                    activities.append(ActivityRecord(
                        timestamp=timestamp,
                        duration=duration,
                        app=row.get('App / Domain', ''),
                        title=row.get('Title', ''),
                        employee=row.get('Employee', '')
                    ))
                except (ValueError, KeyError, IndexError):
                    continue

        return sorted(activities, key=lambda x: x.timestamp)

    def load_keystrokes(self, csv_path: Path) -> List[KeystrokeRecord]:
        """
        Load keystroke records from CSV file.

        Args:
            csv_path: Path to KeystrokeGrid CSV file

        Returns:
            List of KeystrokeRecord objects
        """
        keystrokes = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.strptime(row['Timestamp'], '%Y-%m-%d %H:%M:%S')

                    keystrokes.append(KeystrokeRecord(
                        timestamp=timestamp,
                        app=row.get('App / Domain', ''),
                        keystroke=row.get('Keystroke', ''),
                        employee=row.get('Employee', '')
                    ))
                except (ValueError, KeyError):
                    continue

        return sorted(keystrokes, key=lambda x: x.timestamp)

    def find_active_periods(self, activities: List[ActivityRecord]) -> List[ActivePeriod]:
        """
        Identify continuous active periods from activity records.

        Periods are separated by gaps longer than max_gap_seconds.

        Args:
            activities: List of activity records (should be sorted by timestamp)

        Returns:
            List of ActivePeriod objects
        """
        if not activities:
            return []

        periods = []

        # Start first period
        current_period = ActivePeriod(
            start=activities[0].timestamp,
            end=activities[0].timestamp + timedelta(seconds=activities[0].duration),
            active_time=activities[0].duration,
            activities=[activities[0]]
        )

        # Process remaining activities
        for i in range(1, len(activities)):
            current_activity = activities[i]
            prev_activity = activities[i - 1]

            # Calculate gap between activities
            prev_end = prev_activity.timestamp + timedelta(seconds=prev_activity.duration)
            gap = (current_activity.timestamp - prev_end).total_seconds()

            if gap < self.max_gap_seconds:
                # Continue current period
                current_period.activities.append(current_activity)
                current_period.active_time += current_activity.duration
                current_period.end = current_activity.timestamp + timedelta(seconds=current_activity.duration)
            else:
                # Save current period and start new one
                periods.append(current_period)
                current_period = ActivePeriod(
                    start=current_activity.timestamp,
                    end=current_activity.timestamp + timedelta(seconds=current_activity.duration),
                    active_time=current_activity.duration,
                    activities=[current_activity]
                )

        # Add final period
        periods.append(current_period)

        return periods

    def add_keystrokes_to_periods(self, periods: List[ActivePeriod],
                                  keystrokes: List[KeystrokeRecord]) -> None:
        """
        Add keystroke records to their corresponding active periods.

        Args:
            periods: List of active periods
            keystrokes: List of keystroke records
        """
        for keystroke in keystrokes:
            for period in periods:
                if period.start <= keystroke.timestamp <= period.end:
                    period.keystrokes.append(keystroke)
                    break

    def find_best_match(self, periods: List[ActivePeriod],
                       video_duration: int,
                       tolerance: float = 0.2) -> List[ActivePeriod]:
        """
        Find the best combination of periods that match the video duration.

        Args:
            periods: List of active periods
            video_duration: Video duration in seconds
            tolerance: Acceptable difference as a fraction (default 0.2 = 20%)

        Returns:
            List of periods that best match the video duration
        """
        # Sort periods by active time (longest first)
        sorted_periods = sorted(periods, key=lambda x: x.active_time, reverse=True)

        best_match = []
        best_diff = float('inf')

        # Try single period match
        for period in sorted_periods:
            diff = abs(period.active_time - video_duration)
            if diff < best_diff:
                best_diff = diff
                best_match = [period]

        # Try combinations of top periods
        from itertools import combinations

        for combo_size in range(2, min(6, len(sorted_periods) + 1)):
            for combo in combinations(sorted_periods[:10], combo_size):
                total_time = sum(p.active_time for p in combo)
                diff = abs(total_time - video_duration)

                # Must be within tolerance
                if diff / video_duration <= tolerance and diff < best_diff:
                    best_diff = diff
                    best_match = list(combo)

        # Sort matched periods by start time
        return sorted(best_match, key=lambda x: x.start)

    def match_video_to_activity(self, video_path: Path,
                               activity_csv: Path,
                               keystroke_csv: Optional[Path] = None) -> Optional[VideoMatch]:
        """
        Match a video file to activity and keystroke data.

        Args:
            video_path: Path to video file
            activity_csv: Path to activity CSV file
            keystroke_csv: Optional path to keystroke CSV file

        Returns:
            VideoMatch object, or None if video duration cannot be determined
        """
        # Get video duration
        video_duration = self.get_video_duration(video_path)
        if video_duration is None:
            return None

        # Load activity data
        activities = self.load_activities(activity_csv)

        # Find active periods
        periods = self.find_active_periods(activities)

        # Add keystroke data if available
        if keystroke_csv and keystroke_csv.exists():
            keystrokes = self.load_keystrokes(keystroke_csv)
            self.add_keystrokes_to_periods(periods, keystrokes)

        # Find best matching periods
        matched_periods = self.find_best_match(periods, video_duration)

        # Calculate match accuracy
        total_matched_time = sum(p.active_time for p in matched_periods)
        match_accuracy = 1.0 - abs(total_matched_time - video_duration) / video_duration

        return VideoMatch(
            video_path=video_path,
            video_duration=video_duration,
            matched_periods=matched_periods,
            match_accuracy=match_accuracy
        )

    def generate_report(self, match: VideoMatch) -> str:
        """
        Generate a human-readable report of the video match.

        Args:
            match: VideoMatch object

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"VIDEO MATCH REPORT: {match.video_path.name}")
        lines.append("=" * 80)
        lines.append(f"Video Duration: {match.video_duration // 60}m {match.video_duration % 60}s ({match.video_duration:,} seconds)")
        lines.append(f"Matched Time:   {match.total_matched_time // 60}m {match.total_matched_time % 60}s ({match.total_matched_time:,} seconds)")
        lines.append(f"Difference:     {match.time_difference // 60}m {match.time_difference % 60}s ({match.time_difference:,} seconds)")
        lines.append(f"Match Accuracy: {match.match_accuracy * 100:.1f}%")
        lines.append("")
        lines.append(f"Matched Periods: {len(match.matched_periods)}")
        lines.append("-" * 80)

        for i, period in enumerate(match.matched_periods, 1):
            lines.append(f"\nPeriod {i}:")
            lines.append(f"  Start:        {period.start.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  End:          {period.end.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Timespan:     {period.timespan // 60}m {period.timespan % 60}s")
            lines.append(f"  Active Time:  {period.active_time // 60}m {period.active_time % 60}s ({(period.active_time / period.timespan * 100):.1f}% active)")
            lines.append(f"  Activities:   {len(period.activities)}")
            lines.append(f"  Keystrokes:   {period.get_keystroke_count()}")

            # Top apps
            top_apps = period.get_top_apps(5)
            if top_apps:
                lines.append(f"  Top Apps:")
                for app, duration in top_apps:
                    app_name = app.split('.')[-1] if '.' in app else app
                    lines.append(f"    - {app_name}: {duration // 60}m {duration % 60}s")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def main():
    """Example usage of the VideoActivityMatcher."""
    import sys

    # Set up matcher
    matcher = VideoActivityMatcher(max_gap_seconds=300)

    # Define data paths
    base_dir = Path(__file__).parent.parent / "data"

    sessions = [
        {
            'name': 'Session 1 (sassion1)',
            'video': base_dir / 'sassion1' / 'video.mp4',
            'activity': base_dir / 'sassion1' / 'ActivityGrid #8.csv',
            'keystroke': base_dir / 'sassion1' / 'KeystrokeGrid #4.csv'
        },
        {
            'name': 'Session 2',
            'video': base_dir / 'session2' / 'video.mp4',
            'activity': base_dir / 'session2' / 'ActivityGrid #8.csv',
            'keystroke': base_dir / 'session2' / 'KeystrokeGrid #4.csv'
        }
    ]

    # Process each session
    for session in sessions:
        print(f"\n{'=' * 80}")
        print(f"Processing: {session['name']}")
        print(f"{'=' * 80}")

        if not session['video'].exists():
            print(f"❌ Video not found: {session['video']}")
            continue

        if not session['activity'].exists():
            print(f"❌ Activity data not found: {session['activity']}")
            continue

        # Match video to activity
        match = matcher.match_video_to_activity(
            video_path=session['video'],
            activity_csv=session['activity'],
            keystroke_csv=session['keystroke'] if session['keystroke'].exists() else None
        )

        if match is None:
            print(f"❌ Could not determine video duration")
            continue

        # Generate and print report
        print(matcher.generate_report(match))

        # Export detailed data to JSON
        import json
        output_path = session['video'].parent / f"{session['video'].stem}_match.json"

        match_data = {
            'video_path': str(match.video_path),
            'video_duration_seconds': match.video_duration,
            'total_matched_time_seconds': match.total_matched_time,
            'match_accuracy': match.match_accuracy,
            'periods': []
        }

        for period in match.matched_periods:
            period_data = {
                'start': period.start.isoformat(),
                'end': period.end.isoformat(),
                'timespan_seconds': period.timespan,
                'active_time_seconds': period.active_time,
                'activity_count': len(period.activities),
                'keystroke_count': period.get_keystroke_count(),
                'top_apps': [{'app': app, 'duration_seconds': dur} for app, dur in period.get_top_apps()],
                'activities': [
                    {
                        'timestamp': act.timestamp.isoformat(),
                        'duration_seconds': act.duration,
                        'app': act.app,
                        'title': act.title,
                        'employee': act.employee
                    }
                    for act in period.activities
                ]
            }
            match_data['periods'].append(period_data)

        with open(output_path, 'w') as f:
            json.dump(match_data, f, indent=2)

        print(f"\n✅ Detailed match data saved to: {output_path}")


if __name__ == "__main__":
    main()
