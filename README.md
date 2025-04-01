# Event Attendance Analysis Dashboard

An interactive dashboard for analyzing event attendance data, built with Streamlit.

## Features

- Upload and analyze Excel files containing event attendance data
- Search by attendee name or event name
- View comprehensive attendance statistics and visualizations
- Name matching and deduplication
- Advanced analytics including:
  - Attendance trends
  - Seasonal patterns
  - Correlation analysis
  - Time series decomposition

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the dashboard locally:
```bash
streamlit run event_analysis.py
```

## Data Format

The dashboard expects an Excel file with the following structure:
- Columns containing attendee names
- Columns containing event names
- Date columns for event dates

## Deployment

This app is deployed on Streamlit Cloud. Visit [your-app-url] to access the live version.

## License

MIT License 