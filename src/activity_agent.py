#!/usr/bin/env python3
"""
LangGraph agent for detecting repeating workflows from activity grid data.
Identifies recurring sequences of applications and activities.
"""

import os
import pandas as pd
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Tuple
from collections import Counter
from dotenv import load_dotenv
import anthropic

from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()


class ActivityState(TypedDict):
    """State for the workflow detection agent."""
    csv_path: str
    df: Any  # pandas DataFrame
    keystroke_df: Any  # pandas DataFrame for keystroke data
    video_ocr_data: Any  # OCR data from video frames (optional)
    sequences: List[Tuple[str, ...]]
    workflow_patterns: List[Dict[str, Any]]
    specific_workflows: List[Dict[str, Any]]  # Detailed workflow analysis
    summary: str
    current_step: str


class WorkflowDetectionAgent:
    """LangGraph agent for detecting repeating workflows from activity data."""

    def __init__(self, csv_path: str, keystroke_path: str = None, video_ocr_path: str = None, sequence_length: int = 3):
        self.csv_path = csv_path
        self.keystroke_path = keystroke_path
        self.video_ocr_path = video_ocr_path  # Path to video verification JSON with OCR data
        self.sequence_length = sequence_length  # Length of workflow sequences to detect
        self.client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        self.model = "claude-3-5-haiku-20241022"  # Using Haiku for faster analysis

    def load_data(self, state: ActivityState) -> ActivityState:
        """Load and parse CSV data."""
        print("📂 Loading activity data...")

        try:
            df = pd.read_csv(self.csv_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Duration_seconds'] = pd.to_timedelta(df['Time']).dt.total_seconds()

            # Sort by timestamp
            df = df.sort_values('Timestamp')

            state['df'] = df
            print(f"✓ Loaded {len(df)} activity records")

            # Load keystroke data if available
            if self.keystroke_path and os.path.exists(self.keystroke_path):
                keystroke_df = pd.read_csv(self.keystroke_path)
                keystroke_df['Timestamp'] = pd.to_datetime(keystroke_df['Timestamp'])
                keystroke_df = keystroke_df.sort_values('Timestamp')
                state['keystroke_df'] = keystroke_df
                print(f"✓ Loaded {len(keystroke_df)} keystroke records")
            else:
                state['keystroke_df'] = None

            # Load video OCR data if available
            if self.video_ocr_path and os.path.exists(self.video_ocr_path):
                import json
                with open(self.video_ocr_path) as f:
                    video_ocr_data = json.load(f)
                state['video_ocr_data'] = video_ocr_data
                print(f"✓ Loaded video OCR data ({len(video_ocr_data.get('samples', []))} frames)")
            else:
                state['video_ocr_data'] = None

            state['current_step'] = 'data_loaded'

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            state['current_step'] = 'error'

        return state

    def detect_sequences(self, state: ActivityState) -> ActivityState:
        """Extract activity sequences from the data."""
        print(f"🔍 Detecting sequences of length {self.sequence_length}...")

        try:
            df = state['df']

            # Get sequence of apps/domains
            activities = df['App / Domain'].tolist()

            # Create sliding window sequences
            sequences = []
            for i in range(len(activities) - self.sequence_length + 1):
                seq = tuple(activities[i:i + self.sequence_length])
                sequences.append(seq)

            state['sequences'] = sequences
            state['current_step'] = 'sequences_detected'
            print(f"✓ Extracted {len(sequences)} sequences")

        except Exception as e:
            print(f"❌ Error detecting sequences: {e}")
            state['current_step'] = 'error'

        return state

    def find_patterns(self, state: ActivityState) -> ActivityState:
        """Find repeating workflow patterns in sequences."""
        print("🔄 Finding repeating patterns...")

        try:
            sequences = state['sequences']
            df = state['df']

            # Count sequence frequencies
            sequence_counts = Counter(sequences)

            # Filter to sequences that:
            # 1. Repeat at least twice
            # 2. Involve switching between different apps (not just same app repeated)
            repeating = []
            for seq, count in sequence_counts.items():
                if count >= 2:
                    # Check if sequence involves different apps (indicates actual workflow)
                    unique_apps = len(set(seq))
                    if unique_apps > 1:  # Must have at least 2 different apps
                        repeating.append((seq, count))

            repeating.sort(key=lambda x: x[1], reverse=True)

            # Create workflow patterns with metadata
            workflow_patterns = []
            for seq, count in repeating[:20]:  # Top 20 patterns
                # Find timestamps where this sequence occurs
                occurrences = []
                for i in range(len(state['sequences'])):
                    if state['sequences'][i] == seq:
                        timestamp = df.iloc[i]['Timestamp']
                        occurrences.append(timestamp)

                workflow_patterns.append({
                    'sequence': seq,
                    'count': count,
                    'occurrences': occurrences,
                    'unique_apps': len(set(seq))
                })

            state['workflow_patterns'] = workflow_patterns
            state['current_step'] = 'patterns_found'
            print(f"✓ Found {len(workflow_patterns)} repeating workflow patterns (cross-app)")

        except Exception as e:
            print(f"❌ Error finding patterns: {e}")
            state['current_step'] = 'error'

        return state

    def detect_specific_workflows(self, state: ActivityState) -> ActivityState:
        """Detect specific workflows like copy-paste by analyzing keystrokes, timing, and OCR."""
        print("🔎 Analyzing specific workflow patterns (copy-paste, app switching, OCR content, etc.)...")

        specific_workflows = []

        try:
            df = state['df']
            keystroke_df = state['keystroke_df']

            if keystroke_df is not None:
                # Extract text content from keystrokes (filter out commands)
                def extract_text(keystroke_str):
                    """Extract actual typed text, removing command sequences."""
                    import re
                    # Remove command sequences like <Cmd+C>, <Tab>, etc.
                    text = re.sub(r'<[^>]+>', '', keystroke_str)
                    # Remove excessive spaces
                    text = ' '.join(text.split())
                    return text.strip()

                # Detect repetitive text patterns
                text_patterns = []
                for idx, row in keystroke_df.iterrows():
                    text = extract_text(str(row['Keystroke']))
                    if len(text) > 10:  # Only meaningful text
                        text_patterns.append({
                            'text': text[:100],  # First 100 chars
                            'app': row['App / Domain'],
                            'timestamp': row['Timestamp'],
                            'full_keystroke': str(row['Keystroke'])
                        })

                # Detect repetitive typing (same text multiple times)
                if text_patterns:
                    from collections import Counter as TextCounter
                    text_freq = TextCounter([p['text'] for p in text_patterns])
                    repetitive = [(text, count) for text, count in text_freq.items() if count >= 2]

                    if repetitive:
                        specific_workflows.append({
                            'workflow_type': 'Repetitive Typing',
                            'pattern': f"{len(repetitive)} unique text patterns repeated",
                            'count': sum(count for _, count in repetitive),
                            'examples': [f'"{text}" ({count}x)' for text, count in repetitive[:5]],
                            'description': 'Same text typed multiple times - candidate for text expansion/snippets'
                        })

                # Detect typing-then-copy workflows
                type_copy_patterns = []
                for idx, row in keystroke_df.iterrows():
                    keystroke = str(row['Keystroke'])
                    text = extract_text(keystroke)

                    # Check if this row has both typed text AND copy command
                    if len(text) > 5 and ('<Cmd+C>' in keystroke or '<Cmd+A><Cmd+C>' in keystroke):
                        type_copy_patterns.append({
                            'text': text,
                            'app': row['App / Domain'],
                            'timestamp': row['Timestamp'],
                            'next_action': 'copy'
                        })

                if type_copy_patterns:
                    specific_workflows.append({
                        'workflow_type': 'Type-Then-Copy Pattern',
                        'pattern': f'User types content then immediately copies',
                        'count': len(type_copy_patterns),
                        'examples': [f'{p["text"][:50]} in {p["app"]}' for p in type_copy_patterns[:3]],
                        'description': 'Typing content and immediately copying suggests template/snippet opportunity'
                    })

                # Detect correction patterns (backspace usage)
                correction_patterns = []
                for idx, row in keystroke_df.iterrows():
                    keystroke = str(row['Keystroke'])
                    backspace_count = keystroke.count('<Backspace>')

                    if backspace_count >= 3:  # Multiple corrections
                        text = extract_text(keystroke)
                        correction_patterns.append({
                            'backspaces': backspace_count,
                            'app': row['App / Domain'],
                            'text': text[:50] if text else '[navigation only]',
                            'timestamp': row['Timestamp']
                        })

                if correction_patterns:
                    total_backspaces = sum(p['backspaces'] for p in correction_patterns)
                    specific_workflows.append({
                        'workflow_type': 'Frequent Corrections',
                        'pattern': f'{total_backspaces} backspaces across {len(correction_patterns)} entries',
                        'count': len(correction_patterns),
                        'description': 'High correction rate may indicate typing errors or unfamiliar terms',
                        'examples': [f"{p['backspaces']} backspaces in {p['app']}" for p in correction_patterns[:3]]
                    })

                # Detect search patterns (Return key after text)
                search_patterns = []
                for idx, row in keystroke_df.iterrows():
                    keystroke = str(row['Keystroke'])
                    text = extract_text(keystroke)

                    if len(text) > 3 and '<Return>' in keystroke and 'browser' in row['App / Domain'].lower():
                        search_patterns.append({
                            'query': text,
                            'app': row['App / Domain'],
                            'timestamp': row['Timestamp']
                        })

                if search_patterns:
                    # Find common search terms
                    from collections import Counter as QueryCounter
                    query_freq = QueryCounter([p['query'].lower() for p in search_patterns])
                    repeated_searches = [(q, c) for q, c in query_freq.items() if c >= 2]

                    if repeated_searches:
                        specific_workflows.append({
                            'workflow_type': 'Repeated Searches',
                            'pattern': f'{len(repeated_searches)} queries searched multiple times',
                            'count': sum(c for _, c in repeated_searches),
                            'examples': [f'"{q}" ({c}x)' for q, c in repeated_searches[:5]],
                            'description': 'Repeated searches suggest bookmarking or automation opportunity'
                        })

                # Detect navigation patterns (arrow keys, Cmd+Left/Right)
                navigation_heavy = []
                for idx, row in keystroke_df.iterrows():
                    keystroke = str(row['Keystroke'])
                    nav_count = (keystroke.count('<Up>') + keystroke.count('<Down>') +
                                keystroke.count('<Left>') + keystroke.count('<Right>') +
                                keystroke.count('<Cmd+Left>') + keystroke.count('<Cmd+Right>'))

                    if nav_count >= 10:  # Heavy navigation
                        text = extract_text(keystroke)
                        navigation_heavy.append({
                            'nav_keys': nav_count,
                            'app': row['App / Domain'],
                            'text': text[:50] if text else '[navigation only]'
                        })

                if navigation_heavy:
                    specific_workflows.append({
                        'workflow_type': 'Heavy Navigation',
                        'pattern': f'{len(navigation_heavy)} instances of excessive keyboard navigation',
                        'count': len(navigation_heavy),
                        'description': 'Heavy use of arrow/navigation keys suggests UI inefficiency',
                        'examples': [f"{p['nav_keys']} nav keys in {p['app']}" for p in navigation_heavy[:3]]
                    })

                # Original copy-paste detection
                # Detect copy-paste workflows
                copy_paste_patterns = []

                for idx, row in keystroke_df.iterrows():
                    keystroke = str(row['Keystroke'])

                    # Look for Cmd+C (copy) followed by Cmd+Tab and Cmd+V (paste)
                    if '<Cmd+C>' in keystroke or '<Cmd+X>' in keystroke:
                        # Find next few rows within 30 seconds
                        current_time = row['Timestamp']
                        next_rows = keystroke_df[
                            (keystroke_df['Timestamp'] > current_time) &
                            (keystroke_df['Timestamp'] <= current_time + pd.Timedelta(seconds=30))
                        ].head(10)

                        for _, next_row in next_rows.iterrows():
                            next_keystroke = str(next_row['Keystroke'])
                            if '<Cmd+V>' in next_keystroke:
                                # Found a copy-paste pattern
                                source_app = row['App / Domain']
                                dest_app = next_row['App / Domain']
                                time_diff = (next_row['Timestamp'] - current_time).total_seconds()

                                if source_app != dest_app:  # Cross-app copy-paste
                                    copy_paste_patterns.append({
                                        'type': 'copy-paste',
                                        'source_app': source_app,
                                        'dest_app': dest_app,
                                        'timestamp': current_time,
                                        'duration': time_diff,
                                        'keystroke_sequence': f"{keystroke} → ... → {next_keystroke}"
                                    })
                                break

                # Count frequency of copy-paste patterns
                if copy_paste_patterns:
                    from collections import Counter as CopyCounter
                    pattern_counts = CopyCounter([
                        (p['source_app'], p['dest_app'])
                        for p in copy_paste_patterns
                    ])

                    for (source, dest), count in pattern_counts.most_common(10):
                        examples = [p for p in copy_paste_patterns
                                  if p['source_app'] == source and p['dest_app'] == dest]
                        avg_duration = sum(p['duration'] for p in examples) / len(examples)

                        specific_workflows.append({
                            'workflow_type': 'Copy-Paste Workflow',
                            'pattern': f"{source} → {dest}",
                            'count': count,
                            'avg_duration': avg_duration,
                            'examples': examples[:3]
                        })

                # Detect rapid app switching (Cmd+Tab patterns)
                app_switching = []
                for idx, row in keystroke_df.iterrows():
                    keystroke = str(row['Keystroke'])
                    if '<Cmd+Tab>' in keystroke or keystroke.count('<Cmd+Tab>') > 1:
                        app_switching.append({
                            'timestamp': row['Timestamp'],
                            'app': row['App / Domain'],
                            'switches': keystroke.count('<Cmd+Tab>')
                        })

                if app_switching:
                    specific_workflows.append({
                        'workflow_type': 'Frequent App Switching',
                        'pattern': 'Rapid Cmd+Tab usage',
                        'count': len(app_switching),
                        'total_switches': sum(s['switches'] for s in app_switching),
                        'examples': app_switching[:5]
                    })

            # Detect quick app transitions from activity data
            quick_transitions = []
            for i in range(len(df) - 1):
                duration = df.iloc[i]['Duration_seconds']
                if duration < 5:  # Less than 5 seconds in an app
                    quick_transitions.append({
                        'from_app': df.iloc[i]['App / Domain'],
                        'to_app': df.iloc[i + 1]['App / Domain'],
                        'duration': duration,
                        'timestamp': df.iloc[i]['Timestamp']
                    })

            if quick_transitions:
                from collections import Counter as TransitionCounter
                transition_counts = TransitionCounter([
                    (t['from_app'], t['to_app'])
                    for t in quick_transitions
                ])

                for (from_app, to_app), count in transition_counts.most_common(5):
                    if count >= 3:  # Only show if happens 3+ times
                        specific_workflows.append({
                            'workflow_type': 'Quick App Transitions',
                            'pattern': f"{from_app} → {to_app}",
                            'count': count,
                            'description': 'Quick switches (<5s) suggesting automated or scripted behavior'
                        })

            # Detect workflows from video OCR data
            video_ocr_data = state.get('video_ocr_data')
            if video_ocr_data and 'samples' in video_ocr_data:
                print("📹 Analyzing video OCR content...")

                ocr_samples = video_ocr_data['samples']

                # Detect form-filling workflows
                form_filling_patterns = []
                for sample in ocr_samples:
                    ocr_text = sample.get('ocr_text', '').lower()

                    # Look for form indicators (labels, input fields, buttons)
                    form_indicators = ['name:', 'email:', 'password:', 'address:', 'phone:',
                                     'submit', 'login', 'register', 'sign up', 'form']

                    if any(indicator in ocr_text for indicator in form_indicators):
                        form_filling_patterns.append({
                            'timestamp': sample['real_timestamp'],
                            'app': sample['expected_app'],
                            'ocr_preview': ocr_text[:100],
                            'indicators': [ind for ind in form_indicators if ind in ocr_text]
                        })

                if form_filling_patterns:
                    specific_workflows.append({
                        'workflow_type': 'Form Filling Detected (via OCR)',
                        'pattern': f'User filling out forms in {len(set(p["app"] for p in form_filling_patterns))} different apps',
                        'count': len(form_filling_patterns),
                        'description': 'OCR detected form fields - candidate for auto-fill or form automation',
                        'examples': [f'{p["indicators"]} in {p["app"]}' for p in form_filling_patterns[:3]]
                    })

                # Detect spreadsheet/data entry workflows
                spreadsheet_patterns = []
                for sample in ocr_samples:
                    ocr_text = sample.get('ocr_text', '').lower()
                    app = sample.get('expected_app', '').lower()

                    # Excel/spreadsheet indicators
                    if 'excel' in app or any(word in ocr_text for word in ['format as table', 'conditional formatting', 'cells', 'sheet']):
                        spreadsheet_patterns.append({
                            'timestamp': sample['real_timestamp'],
                            'app': sample['expected_app'],
                            'ocr_preview': ocr_text[:80]
                        })

                if spreadsheet_patterns:
                    specific_workflows.append({
                        'workflow_type': 'Spreadsheet Data Work (via OCR)',
                        'pattern': f'Working with spreadsheets/tables',
                        'count': len(spreadsheet_patterns),
                        'description': 'OCR shows spreadsheet work - consider Excel macros or Python automation'
                    })

                # Detect repeated URL patterns
                url_patterns = []
                from collections import Counter as URLCounter
                for sample in ocr_samples:
                    ocr_text = sample.get('ocr_text', '')

                    # Simple URL detection
                    import re
                    urls = re.findall(r'https?://[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:\.[a-zA-Z]{2,})', ocr_text)

                    for url in urls:
                        url_patterns.append({
                            'url': url.lower(),
                            'timestamp': sample['real_timestamp'],
                            'app': sample['expected_app']
                        })

                if url_patterns:
                    url_freq = URLCounter([p['url'] for p in url_patterns])
                    repeated_urls = [(url, count) for url, count in url_freq.items() if count >= 2]

                    if repeated_urls:
                        specific_workflows.append({
                            'workflow_type': 'Repeated Website Visits (via OCR)',
                            'pattern': f'{len(repeated_urls)} websites visited multiple times',
                            'count': sum(count for _, count in repeated_urls),
                            'examples': [f'{url} ({count}x)' for url, count in repeated_urls[:5]],
                            'description': 'Repeatedly visiting same URLs - bookmark or automate navigation'
                        })

                # Detect document/text editing workflows
                text_editing_patterns = []
                for sample in ocr_samples:
                    ocr_text = sample.get('ocr_text', '').lower()
                    app = sample.get('expected_app', '').lower()

                    # Text editor indicators
                    if any(word in app for word in ['textedit', 'word', 'docs', 'notion']):
                        # Look for document structure
                        if any(word in ocr_text for word in ['title', 'heading', 'bullet', 'paragraph', 'document']):
                            text_editing_patterns.append({
                                'timestamp': sample['real_timestamp'],
                                'app': sample['expected_app'],
                                'title': sample.get('expected_title', '')
                            })

                if text_editing_patterns:
                    specific_workflows.append({
                        'workflow_type': 'Document Editing (via OCR)',
                        'pattern': f'Editing documents',
                        'count': len(text_editing_patterns),
                        'description': 'Document editing detected - consider templates or snippets'
                    })

                # Detect calendar/scheduling workflows
                calendar_patterns = []
                for sample in ocr_samples:
                    ocr_text = sample.get('ocr_text', '').lower()
                    app = sample.get('expected_app', '').lower()

                    if 'calendar' in app or any(word in ocr_text for word in ['schedule', 'meeting', 'appointment', 'event']):
                        calendar_patterns.append({
                            'timestamp': sample['real_timestamp'],
                            'app': sample['expected_app']
                        })

                if calendar_patterns:
                    specific_workflows.append({
                        'workflow_type': 'Calendar/Scheduling (via OCR)',
                        'pattern': f'Managing calendar events',
                        'count': len(calendar_patterns),
                        'description': 'Calendar management detected - automate meeting scheduling'
                    })

                # Detect credential/password entry
                credential_patterns = []
                for sample in ocr_samples:
                    ocr_text = sample.get('ocr_text', '')

                    # Look for password/credential indicators
                    if any(word in ocr_text.lower() for word in ['password', 'login', 'username', 'id:', 'pass:']):
                        credential_patterns.append({
                            'timestamp': sample['real_timestamp'],
                            'app': sample['expected_app'],
                            'has_visible_password': 'pass:' in ocr_text.lower() or 'password:' in ocr_text.lower()
                        })

                if credential_patterns:
                    specific_workflows.append({
                        'workflow_type': 'Credential Entry (via OCR)',
                        'pattern': f'Entering credentials/passwords',
                        'count': len(credential_patterns),
                        'description': 'Multiple login/credential entries - use password manager or SSO',
                        'security_note': '⚠️ Some passwords may be visible in OCR data'
                    })

            state['specific_workflows'] = specific_workflows
            state['current_step'] = 'specific_detected'
            print(f"✓ Found {len(specific_workflows)} specific workflow patterns")

        except Exception as e:
            print(f"❌ Error detecting specific workflows: {e}")
            state['specific_workflows'] = []
            state['current_step'] = 'specific_detected'

        return state

    def analyze_workflows(self, state: ActivityState) -> ActivityState:
        """Use Claude to analyze and interpret the workflow patterns."""
        print("📝 Analyzing workflow patterns with AI...")

        if not state['workflow_patterns']:
            state['summary'] = "No repeating workflow patterns found."
            state['current_step'] = 'completed'
            return state

        # Prepare patterns summary for Claude
        patterns_text = ""
        for i, pattern in enumerate(state['workflow_patterns'][:10], 1):
            seq_str = " → ".join(pattern['sequence'])
            patterns_text += f"\n{i}. Workflow: {seq_str}\n"
            patterns_text += f"   Repeated: {pattern['count']} times\n"
            patterns_text += f"   Occurrences: {', '.join([str(t) for t in pattern['occurrences'][:3]])}"
            if len(pattern['occurrences']) > 3:
                patterns_text += f" ... and {len(pattern['occurrences']) - 3} more"
            patterns_text += "\n"

        prompt = f"""Analyze these repeating workflow patterns from activity tracking data. These are sequences where users switch between different applications, suggesting manual workflows that could potentially be automated:

{patterns_text}

For each pattern:
1. Describe what the workflow likely represents (e.g., "Copy from browser to email", "Data entry workflow", "Research and documentation")
2. Identify if this looks like a manual process that could be automated
3. Suggest specific automation opportunities (e.g., "Use clipboard manager", "Create a script to...", "Use RPA tools")
4. Estimate potential time savings if automated

Focus on practical automation suggestions for repetitive cross-application workflows."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            state['summary'] = message.content[0].text
            state['current_step'] = 'completed'
            print("✓ Workflow analysis completed")

        except Exception as e:
            print(f"❌ Error analyzing workflows: {e}")
            state['summary'] = f"Error analyzing workflows: {str(e)}"
            state['current_step'] = 'error'

        return state

    def should_continue(self, state: ActivityState) -> str:
        """Determine next step based on current state."""
        if state['current_step'] == 'error':
            return 'end'
        elif state['current_step'] == 'data_loaded':
            return 'detect'
        elif state['current_step'] == 'sequences_detected':
            return 'find'
        elif state['current_step'] == 'patterns_found':
            return 'detect_specific'
        elif state['current_step'] == 'specific_detected':
            return 'analyze'
        elif state['current_step'] == 'completed':
            return 'end'
        return 'end'

    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ActivityState)

        # Add nodes
        workflow.add_node("load", self.load_data)
        workflow.add_node("detect", self.detect_sequences)
        workflow.add_node("find", self.find_patterns)
        workflow.add_node("detect_specific", self.detect_specific_workflows)
        workflow.add_node("analyze", self.analyze_workflows)

        # Set entry point
        workflow.set_entry_point("load")

        # Add conditional edges
        workflow.add_conditional_edges(
            "load",
            self.should_continue,
            {
                "detect": "detect",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "detect",
            self.should_continue,
            {
                "find": "find",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "find",
            self.should_continue,
            {
                "detect_specific": "detect_specific",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "detect_specific",
            self.should_continue,
            {
                "analyze": "analyze",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "analyze",
            self.should_continue,
            {
                "end": END
            }
        )

        return workflow.compile()

    def analyze(self) -> Dict[str, Any]:
        """Run the complete workflow detection."""
        print("\n" + "="*60)
        print("🔄 Repeating Workflow Detection Agent")
        print("="*60 + "\n")

        # Initialize state
        initial_state = ActivityState(
            csv_path=self.csv_path,
            df=None,
            keystroke_df=None,
            video_ocr_data=None,
            sequences=[],
            workflow_patterns=[],
            specific_workflows=[],
            summary="",
            current_step="initialized"
        )

        # Build and run the graph
        graph = self.build_graph()
        final_state = graph.invoke(initial_state)

        return final_state


def main():
    """Main entry point for running the agent."""
    # Try to find the most recent ActivityGrid CSV file
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'session2')

    # Look for ActivityGrid files
    import glob
    activity_files = glob.glob(os.path.join(data_dir, 'ActivityGrid*.csv'))

    if not activity_files:
        print(f"❌ Error: No ActivityGrid CSV files found in {data_dir}")
        return

    # Use the most recent one (by filename)
    csv_path = sorted(activity_files)[-1]
    print(f"📁 Using activity file: {os.path.basename(csv_path)}")

    # Look for KeystrokeGrid files
    keystroke_files = glob.glob(os.path.join(data_dir, 'KeystrokeGrid*.csv'))
    keystroke_path = sorted(keystroke_files)[-1] if keystroke_files else None
    if keystroke_path:
        print(f"📁 Using keystroke file: {os.path.basename(keystroke_path)}")

    # Look for video verification JSON (with OCR data)
    video_ocr_files = glob.glob(os.path.join(data_dir, '*verification.json'))
    video_ocr_path = sorted(video_ocr_files)[-1] if video_ocr_files else None
    if video_ocr_path:
        print(f"📹 Using video OCR data: {os.path.basename(video_ocr_path)}")

    # Create and run the agent (detect sequences of 3 activities)
    agent = WorkflowDetectionAgent(csv_path, keystroke_path, video_ocr_path, sequence_length=3)
    result = agent.analyze()

    # Display results
    print("\n" + "="*60)
    print("🔄 REPEATING WORKFLOW PATTERNS")
    print("="*60 + "\n")

    if result['workflow_patterns']:
        print(f"Found {len(result['workflow_patterns'])} repeating workflow patterns:\n")

        for i, pattern in enumerate(result['workflow_patterns'][:10], 1):
            seq_str = " → ".join(pattern['sequence'])
            print(f"{i}. {seq_str}")
            print(f"   Repeated: {pattern['count']} times")
            print()

    # Display specific workflows
    if result.get('specific_workflows'):
        print("\n" + "="*60)
        print("🎯 SPECIFIC WORKFLOW DETAILS")
        print("="*60 + "\n")

        for workflow in result['specific_workflows']:
            print(f"📌 {workflow['workflow_type']}")
            print(f"   Pattern: {workflow['pattern']}")
            print(f"   Occurrences: {workflow['count']}")
            if 'avg_duration' in workflow:
                print(f"   Avg Duration: {workflow['avg_duration']:.2f}s")
            if 'total_switches' in workflow:
                print(f"   Total Switches: {workflow['total_switches']}")
            if 'description' in workflow:
                print(f"   {workflow['description']}")
            if 'security_note' in workflow:
                print(f"   {workflow['security_note']}")
            if 'examples' in workflow:
                print(f"   Examples:")
                for example in workflow['examples'][:5]:
                    if isinstance(example, str):
                        print(f"      • {example}")
                    elif isinstance(example, dict):
                        if 'text' in example:
                            print(f"      • {example['text'][:80]} (in {example.get('app', 'unknown')})")
                        elif 'source_app' in example:
                            print(f"      • {example['source_app']} → {example['dest_app']} ({example['duration']:.1f}s)")
            print()

    print("="*60)
    print("🤖 AI ANALYSIS")
    print("="*60 + "\n")
    print(result['summary'])
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
