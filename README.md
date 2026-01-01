# Activity Monitor

A LangGraph-powered agent that analyzes activity tracking data to extract meaningful patterns and productivity insights.

## Overview

This project uses a LangGraph agent to analyze time tracking data from ActivityGrid CSV files. The agent identifies patterns in application usage, time distribution, productivity trends, and provides actionable insights.

## Features

- **Automated Pattern Extraction**: Uses Claude AI to identify usage patterns, peak productivity hours, and activity categories
- **Multi-Step Analysis**: LangGraph workflow that loads data, extracts patterns, and generates summaries
- **Comprehensive Insights**: Analyzes time usage, application switching, productivity patterns, and anomalies
- **Actionable Recommendations**: Provides concrete suggestions for improving time management

## Project Structure

```
activity_monitor/
├── src/
│   ├── __init__.py
│   └── activity_agent.py      # Main LangGraph agent implementation
├── tests/
│   └── test_api_key.py        # API key validation test
├── data/
│   └── ActivityGrid #9.csv    # Activity tracking data
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── .gitignore
└── README.md
```

## Setup

1. **Clone the repository**
   ```bash
   cd activity_monitor
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the root directory:
   ```
   CLAUDE_API_KEY=your_api_key_here
   ```

5. **Test your API key**
   ```bash
   python tests/test_api_key.py
   ```

## Usage

Run the activity analysis agent:

```bash
python main.py
```

Or run directly:

```bash
python src/activity_agent.py
```

## How It Works

The agent follows a three-step workflow:

1. **Load Data**: Reads and parses the ActivityGrid CSV file, extracting timestamps, applications, durations, and creates a statistical summary

2. **Extract Patterns**: Uses Claude AI to analyze the data and identify:
   - Time usage patterns and active hours
   - Most frequently used applications and websites
   - Productivity insights and focus patterns
   - Context switching frequency
   - Activity categorization
   - Anomalies and interesting observations

3. **Generate Summary**: Creates a concise executive summary with key findings and actionable recommendations

## Data Format

The agent expects CSV files with the following columns:
- `Timestamp`: When the activity occurred
- `Computer`: Computer identifier
- `App / Domain`: Application name or domain
- `Time`: Duration (HH:MM:SS format)
- `Title`: Window title or page title

## Output

The agent provides:
- Detailed pattern analysis across multiple dimensions
- Top 3 key findings
- 2-3 actionable recommendations for productivity improvement
- Statistics on time usage and application distribution

## Requirements

- Python 3.8+
- Anthropic API key with Claude access
- Required packages listed in [requirements.txt](requirements.txt)

## Development

To add new analysis capabilities:

1. Extend the `ActivityState` TypedDict in [activity_agent.py](src/activity_agent.py)
2. Add new nodes to the workflow
3. Update the graph connections in `build_graph()`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
