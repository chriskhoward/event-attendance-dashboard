import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import calendar
import os
from scipy import stats
from fuzzywuzzy import fuzz
from nameparser import HumanName
import re

# Set page config
st.set_page_config(page_title="Event Attendance Analysis", layout="wide")

# Title
st.title("Event Attendance Analysis Dashboard")

# Initialize session state for the dataframe and unique names
if 'df' not in st.session_state:
    st.session_state.df = None
if 'unique_names' not in st.session_state:
    st.session_state.unique_names = []
if 'unique_events' not in st.session_state:
    st.session_state.unique_events = []
if 'selected_events' not in st.session_state:
    st.session_state.selected_events = []
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'name_matches' not in st.session_state:
    st.session_state.name_matches = []

# Name standardization function
def standardize_name(name):
    if pd.isna(name):
        return ""
    # Convert to string and clean
    name = str(name).strip()
    # Parse the name
    parsed = HumanName(name)
    # Get first name and standardize common variations
    first_name = parsed.first.lower()
    # Common first name variations
    name_variations = {
        'christopher': 'chris',
        'robert': 'bob',
        'william': 'bill',
        'james': 'jim',
        'james': 'jimmy',
        'michael': 'mike',
        'joseph': 'joe',
        'thomas': 'tom',
        'thomas': 'tommy',
        'richard': 'dick',
        'richard': 'rick',
        'charles': 'chuck',
        'charles': 'charlie',
        'daniel': 'dan',
        'daniel': 'danny',
        'edward': 'ed',
        'edward': 'eddie',
        'george': 'georgie',
        'henry': 'hank',
        'john': 'johnny',
        'john': 'jack',
        'lawrence': 'larry',
        'matthew': 'matt',
        'nicholas': 'nick',
        'nicholas': 'nicky',
        'patrick': 'pat',
        'patrick': 'paddy',
        'peter': 'pete',
        'peter': 'petey',
        'ronald': 'ron',
        'ronald': 'ronnie',
        'samuel': 'sam',
        'samuel': 'sammy',
        'stephen': 'steve',
        'stephen': 'stevie',
        'theodore': 'ted',
        'theodore': 'teddy',
        'timothy': 'tim',
        'timothy': 'timmy',
        'walter': 'walt',
        'walter': 'wally'
    }
    # Return standardized first name if it exists in variations, otherwise return original
    return name_variations.get(first_name, first_name)

# Function to find similar names
def find_similar_names(df, name_column, email_column=None, threshold=85):
    similar_names = []
    
    # Get unique names
    unique_names = df[name_column].unique()
    
    # Create a dictionary to store standardized names
    name_dict = {name: standardize_name(name) for name in unique_names if pd.notna(name)}
    
    # Compare each name with every other name
    for i, name1 in enumerate(unique_names):
        if pd.isna(name1):
            continue
            
        std_name1 = name_dict[name1]
        
        for name2 in unique_names[i+1:]:
            if pd.isna(name2):
                continue
                
            std_name2 = name_dict[name2]
            
            # Calculate similarity scores
            ratio = fuzz.ratio(std_name1, std_name2)
            partial_ratio = fuzz.partial_ratio(std_name1, std_name2)
            
            # If email column exists, check if emails match
            email_match = False
            if email_column and email_column in df.columns:
                email1 = df[df[name_column] == name1][email_column].iloc[0] if not df[df[name_column] == name1].empty else None
                email2 = df[df[name_column] == name2][email_column].iloc[0] if not df[df[name_column] == name2].empty else None
                email_match = pd.notna(email1) and pd.notna(email2) and email1 == email2
            
            # If either ratio is above threshold or emails match
            if ratio >= threshold or partial_ratio >= threshold or email_match:
                similar_names.append({
                    'name1': name1,
                    'name2': name2,
                    'ratio': ratio,
                    'partial_ratio': partial_ratio,
                    'email_match': email_match,
                    'confidence': max(ratio, partial_ratio) if not email_match else 100
                })
    
    return similar_names

# Function to merge similar records
def merge_similar_records(df, matches, name_column, email_column=None):
    merged_df = df.copy()
    
    # Create a mapping of names to their standardized versions
    name_mapping = {}
    for match in matches:
        if match['confidence'] >= 85:  # Only merge high confidence matches
            name_mapping[match['name2']] = match['name1']
    
    # Apply the mapping
    merged_df[name_column] = merged_df[name_column].map(lambda x: name_mapping.get(x, x))
    
    # Group by the standardized name and aggregate
    if email_column and email_column in merged_df.columns:
        merged_df = merged_df.groupby([name_column, email_column]).agg({
            col: lambda x: '; '.join(x.unique()) if x.dtype == 'object' else x.iloc[0]
            for col in merged_df.columns
            if col not in [name_column, email_column]
        }).reset_index()
    else:
        merged_df = merged_df.groupby(name_column).agg({
            col: lambda x: '; '.join(x.unique()) if x.dtype == 'object' else x.iloc[0]
            for col in merged_df.columns
            if col != name_column
        }).reset_index()
    
    return merged_df

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    # Convert date columns to datetime if they aren't already
    date_columns = df.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            # Try to parse with common date formats
            df[col] = pd.to_datetime(df[col], format='mixed')
        except:
            continue
    return df

# Function to get unique names from the dataframe
def get_unique_names(df):
    names = set()
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            # Get unique values from this column
            unique_values = df[col].dropna().unique()
            # Add to names set if they look like names (contain only letters and spaces)
            names.update([str(val) for val in unique_values if str(val).replace(' ', '').isalpha()])
    return sorted(list(names))

# Function to get unique events from the dataframe
def get_unique_events(df):
    events = set()
    # Look for columns that might contain event names (non-date, non-name columns)
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            # Skip columns that look like they contain names
            if not df[col].astype(str).str.replace(' ', '').str.isalpha().all():
                unique_values = df[col].dropna().unique()
                events.update([str(val) for val in unique_values if len(str(val)) > 0])
    return sorted(list(events))

# Function to calculate event statistics
def calculate_event_stats(event_data, event_dates):
    stats = {}
    
    # Basic counts
    stats['total_attendees'] = len(event_data)
    stats['occurrences'] = len(event_dates)
    stats['avg_attendance'] = stats['total_attendees'] / stats['occurrences'] if stats['occurrences'] > 0 else 0
    
    # Peak attendance
    attendance_by_date = event_data.groupby(event_dates).size()
    stats['peak_attendance'] = attendance_by_date.max()
    stats['peak_date'] = attendance_by_date.idxmax()
    
    # Attendance trend
    if len(attendance_by_date) > 1:
        x = np.arange(len(attendance_by_date))
        slope, _, r_value, _, _ = stats.linregress(x, attendance_by_date.values)
        stats['attendance_trend'] = 'Increasing' if slope > 0 else 'Decreasing'
        stats['trend_strength'] = abs(r_value)
    else:
        stats['attendance_trend'] = 'No trend'
        stats['trend_strength'] = 0
    
    # Consistency
    stats['attendance_std'] = attendance_by_date.std()
    stats['consistency'] = 'High' if stats['attendance_std'] < stats['avg_attendance'] * 0.3 else 'Medium' if stats['attendance_std'] < stats['avg_attendance'] * 0.6 else 'Low'
    
    return stats

# Search section - always visible
st.sidebar.header("Search")

# Try to load default file first
default_file = "2025-03-31 Events HAULYP.xls"
if os.path.exists(default_file):
    if st.session_state.df is None:
        try:
            st.session_state.df = load_data(default_file)
            st.session_state.unique_names = get_unique_names(st.session_state.df)
            st.session_state.unique_events = get_unique_events(st.session_state.df)
        except Exception as e:
            st.error(f"Error loading default file: {str(e)}")
            st.session_state.df = None

# Optional file upload section
st.sidebar.header("Upload Different Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose a different Excel file", type=['xls', 'xlsx'])

if uploaded_file is not None:
    try:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.unique_names = get_unique_names(st.session_state.df)
        st.session_state.unique_events = get_unique_events(st.session_state.df)
    except Exception as e:
        st.error(f"Error loading uploaded file: {str(e)}")
        if st.session_state.df is None:
            st.info("Please make sure the Excel file is in the correct format and contains the expected columns.")
            st.stop()

if st.session_state.df is not None:
    df = st.session_state.df
    
    # Display data preview and column info
    with st.sidebar.expander("View Data Structure"):
        st.write("Columns in dataset:")
        for col in df.columns:
            st.write(f"- {col} ({df[col].dtype})")
    
    # Add search type selector
    search_type = st.sidebar.radio(
        "Search by:",
        ["Attendee Name", "Event Name"]
    )
    
    # Initialize search variables
    search_name = ""
    search_event = ""
    
    if search_type == "Attendee Name":
        # Add search box with autocomplete for names
        search_name = st.sidebar.selectbox(
            "Enter or select attendee name:",
            options=[""] + st.session_state.unique_names,
            format_func=lambda x: "Type to search..." if x == "" else x
        ).strip()
    else:  # Event Name search
        # Add search box with autocomplete for events
        search_event = st.sidebar.selectbox(
            "Enter or select event name:",
            options=[""] + st.session_state.unique_events,
            format_func=lambda x: "Type to search..." if x == "" else x
        ).strip()
    
    if not search_name and not search_event:
        st.info("Please select or type a name or event to see the analysis")
    else:
        if search_type == "Attendee Name":
            # Find the name column (try to identify it)
            name_column = None
            for col in df.columns:
                # Check if column contains string values
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    # Check if any values in this column contain the search name
                    if df[col].astype(str).str.contains(search_name, case=False, na=False).any():
                        name_column = col
                        break
            
            if name_column is None:
                st.warning("Could not find a column containing names. Please check the data structure.")
            else:
                # Filter data for the searched name
                attendee_data = df[df[name_column].astype(str).str.contains(search_name, case=False, na=False)]
                
                if not attendee_data.empty:
                    st.header(f"Analysis for {search_name}")
                    
                    # Create three columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_events = len(attendee_data)
                        st.metric("Total Events Attended", total_events)
                    
                    with col2:
                        unique_dates = attendee_data.select_dtypes(include=['datetime64']).nunique().sum()
                        st.metric("Unique Dates", unique_dates)
                    
                    with col3:
                        # Calculate most common day of week
                        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
                        most_common_day = all_dates.dt.day_name().mode().iloc[0]
                        st.metric("Most Common Day", most_common_day)
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4 = st.tabs(["Event Timeline", "Day of Week Distribution", "Monthly Attendance", "Attendance Patterns"])
                    
                    with tab1:
                        # Create timeline of events
                        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
                        all_dates = all_dates.dropna()
                        
                        # Create a more detailed timeline
                        fig_timeline = go.Figure()
                        
                        # Add scatter plot for events
                        fig_timeline.add_trace(go.Scatter(
                            x=all_dates,
                            y=[1] * len(all_dates),
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='#1f77b4',
                                symbol='circle'
                            ),
                            name='Events'
                        ))
                        
                        # Update layout
                        fig_timeline.update_layout(
                            title="Event Timeline",
                            xaxis_title="Date",
                            yaxis_title="",
                            showlegend=False,
                            height=400,
                            yaxis=dict(showticklabels=False),
                            hovermode='x unified'
                        )
                        
                        # Add hover text
                        fig_timeline.update_traces(
                            hovertemplate="Date: %{x}<br>Event<extra></extra>"
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    with tab2:
                        # Day of week distribution
                        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
                        all_dates = all_dates.dropna()
                        day_dist = all_dates.dt.day_name().value_counts()
                        
                        # Reorder days of week
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_dist = day_dist.reindex(day_order)
                        
                        fig_days = px.bar(
                            x=day_dist.index,
                            y=day_dist.values,
                            title="Attendance by Day of Week",
                            color=day_dist.values,
                            color_continuous_scale='Viridis'
                        )
                        
                        fig_days.update_layout(
                            xaxis_title="Day of Week",
                            yaxis_title="Number of Events",
                            height=400
                        )
                        
                        st.plotly_chart(fig_days, use_container_width=True)
                    
                    with tab3:
                        # Monthly attendance
                        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
                        all_dates = all_dates.dropna()
                        monthly_counts = all_dates.dt.to_period('M').value_counts().sort_index()
                        
                        fig_monthly = px.bar(
                            x=monthly_counts.index.astype(str),
                            y=monthly_counts.values,
                            title="Monthly Attendance",
                            color=monthly_counts.values,
                            color_continuous_scale='Viridis'
                        )
                        
                        fig_monthly.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Number of Events",
                            height=400
                        )
                        
                        st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    with tab4:
                        # Create a heatmap of attendance patterns
                        all_dates = pd.concat([attendee_data[col] for col in attendee_data.select_dtypes(include=['datetime64']).columns])
                        all_dates = all_dates.dropna()
                        
                        # Create a DataFrame with day of week and month
                        attendance_patterns = pd.DataFrame({
                            'date': all_dates,
                            'day_of_week': all_dates.dt.day_name(),
                            'month': all_dates.dt.month_name()
                        })
                        
                        # Create pivot table for heatmap
                        pivot_table = pd.pivot_table(
                            attendance_patterns,
                            values='date',
                            index='day_of_week',
                            columns='month',
                            aggfunc='count'
                        ).fillna(0)
                        
                        # Reorder days of week
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        pivot_table = pivot_table.reindex(day_order)
                        
                        # Create heatmap
                        fig_heatmap = px.imshow(
                            pivot_table,
                            title="Attendance Patterns by Day and Month",
                            aspect='auto',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig_heatmap.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=500
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Display raw data with better formatting
                    st.subheader("Event Details")
                    st.dataframe(
                        attendee_data,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                else:
                    st.warning(f"No data found for {search_name}")
        else:  # Event Name search
            # Find the event column (try to identify it)
            event_column = None
            for col in df.columns:
                # Check if column contains string values
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    # Check if any values in this column contain the search event
                    if df[col].astype(str).str.contains(search_event, case=False, na=False).any():
                        event_column = col
                        break
            
            if event_column is None:
                st.warning("Could not find a column containing events. Please check the data structure.")
            else:
                # Filter data for the searched event
                event_data = df[df[event_column].astype(str).str.contains(search_event, case=False, na=False)]
                
                if not event_data.empty:
                    st.header(f"Analysis for {search_event}")
                    
                    # Get event dates
                    event_dates = event_data.select_dtypes(include=['datetime64']).iloc[:, 0]
                    
                    # Calculate comprehensive statistics
                    stats = calculate_event_stats(event_data, event_dates)
                    
                    # Create three columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Attendees", stats['total_attendees'])
                        st.metric("Average Attendance", f"{stats['avg_attendance']:.1f}")
                    
                    with col2:
                        st.metric("Number of Occurrences", stats['occurrences'])
                        st.metric("Peak Attendance", stats['peak_attendance'])
                    
                    with col3:
                        st.metric("Attendance Trend", stats['attendance_trend'])
                        st.metric("Consistency", stats['consistency'])
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Attendance Over Time", 
                        "Attendee List", 
                        "Event Details",
                        "Event Comparison",
                        "Advanced Analytics"
                    ])
                    
                    with tab1:
                        # Create attendance timeline
                        attendance_counts = event_data.groupby(event_dates).size()
                        
                        fig_timeline = go.Figure()
                        
                        # Add line plot
                        fig_timeline.add_trace(go.Scatter(
                            x=attendance_counts.index,
                            y=attendance_counts.values,
                            mode='lines+markers',
                            name='Attendance'
                        ))
                        
                        # Add trend line
                        if len(attendance_counts) > 1:
                            x = np.arange(len(attendance_counts))
                            z = np.polyfit(x, attendance_counts.values, 1)
                            p = np.poly1d(z)
                            fig_timeline.add_trace(go.Scatter(
                                x=attendance_counts.index,
                                y=p(x),
                                mode='lines',
                                name='Trend',
                                line=dict(dash='dash')
                            ))
                        
                        fig_timeline.update_layout(
                            title="Attendance Over Time",
                            xaxis_title="Date",
                            yaxis_title="Number of Attendees",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Add attendance statistics
                        st.subheader("Attendance Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("Peak Date:", stats['peak_date'].strftime('%Y-%m-%d'))
                            st.write("Standard Deviation:", f"{stats['attendance_std']:.1f}")
                        
                        with col2:
                            st.write("Trend Strength:", f"{stats['trend_strength']:.2f}")
                            st.write("Consistency Level:", stats['consistency'])
                        
                        with col3:
                            st.write("Growth Rate:", f"{stats['attendance_trend']}")
                            st.write("Total Unique Attendees:", len(event_data[event_column].unique()))
                    
                    with tab2:
                        # Get unique attendees for this event
                        name_column = None
                        for col in df.columns:
                            if df[col].dtype == 'object' or df[col].dtype == 'string':
                                if df[col].astype(str).str.replace(' ', '').str.isalpha().all():
                                    name_column = col
                                    break
                        
                        if name_column:
                            unique_attendees = event_data[name_column].unique()
                            st.subheader("Attendees")
                            
                            # Add attendance frequency for each attendee
                            attendee_freq = event_data[name_column].value_counts()
                            
                            # Create a DataFrame with attendee statistics
                            attendee_stats = pd.DataFrame({
                                'Attendee': attendee_freq.index,
                                'Times Attended': attendee_freq.values,
                                'Attendance Rate': (attendee_freq.values / stats['occurrences'] * 100).round(1)
                            })
                            
                            # Display as a table
                            st.dataframe(
                                attendee_stats,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.warning("Could not identify the attendee column")
                    
                    with tab3:
                        # Display event details
                        st.subheader("Event Details")
                        st.dataframe(
                            event_data,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with tab4:
                        st.subheader("Compare with Other Events")
                        
                        # Add event selection for comparison
                        compare_event = st.selectbox(
                            "Select event to compare with:",
                            options=[""] + [e for e in st.session_state.unique_events if e != search_event],
                            format_func=lambda x: "Select an event..." if x == "" else x
                        )
                        
                        if compare_event:
                            # Find and filter data for comparison event
                            compare_data = df[df[event_column].astype(str).str.contains(compare_event, case=False, na=False)]
                            compare_dates = compare_data.select_dtypes(include=['datetime64']).iloc[:, 0]
                            compare_stats = calculate_event_stats(compare_data, compare_dates)
                            
                            # Create comparison metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Attendance Difference",
                                    f"{stats['avg_attendance'] - compare_stats['avg_attendance']:.1f}",
                                    f"{((stats['avg_attendance'] - compare_stats['avg_attendance']) / compare_stats['avg_attendance'] * 100):.1f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Consistency Comparison",
                                    f"{stats['consistency']} vs {compare_stats['consistency']}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Trend Comparison",
                                    f"{stats['attendance_trend']} vs {compare_stats['attendance_trend']}"
                                )
                            
                            # Create comparison visualization
                            fig_compare = go.Figure()
                            
                            # Add traces for both events
                            fig_compare.add_trace(go.Scatter(
                                x=event_dates,
                                y=event_data.groupby(event_dates).size(),
                                name=search_event,
                                mode='lines+markers'
                            ))
                            
                            fig_compare.add_trace(go.Scatter(
                                x=compare_dates,
                                y=compare_data.groupby(compare_dates).size(),
                                name=compare_event,
                                mode='lines+markers'
                            ))
                            
                            fig_compare.update_layout(
                                title="Attendance Comparison",
                                xaxis_title="Date",
                                yaxis_title="Number of Attendees",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_compare, use_container_width=True)
                    
                    with tab5:
                        st.subheader("Advanced Analytics")
                        
                        # Create a DataFrame for advanced analysis
                        analysis_df = pd.DataFrame({
                            'Date': event_dates,
                            'Attendance': event_data.groupby(event_dates).size(),
                            'Day of Week': event_dates.dt.day_name(),
                            'Month': event_dates.dt.month_name()
                        })
                        
                        # Calculate correlations
                        st.write("Correlation Analysis")
                        correlation_matrix = analysis_df[['Attendance', 'Day of Week', 'Month']].corr()
                        fig_corr = px.imshow(
                            correlation_matrix,
                            title="Correlation Matrix",
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Time series decomposition
                        st.write("Time Series Analysis")
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        
                        # Resample data to monthly frequency
                        monthly_data = analysis_df.set_index('Date')['Attendance'].resample('M').mean()
                        
                        # Perform decomposition
                        decomposition = seasonal_decompose(monthly_data, period=12)
                        
                        # Plot components
                        fig_decomp = go.Figure()
                        
                        fig_decomp.add_trace(go.Scatter(
                            x=decomposition.trend.index,
                            y=decomposition.trend,
                            name='Trend'
                        ))
                        
                        fig_decomp.add_trace(go.Scatter(
                            x=decomposition.seasonal.index,
                            y=decomposition.seasonal,
                            name='Seasonal'
                        ))
                        
                        fig_decomp.add_trace(go.Scatter(
                            x=decomposition.resid.index,
                            y=decomposition.resid,
                            name='Residual'
                        ))
                        
                        fig_decomp.update_layout(
                            title="Time Series Decomposition",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            height=600
                        )
                        
                        st.plotly_chart(fig_decomp, use_container_width=True)
                    
                else:
                    st.warning(f"No data found for {search_event}")

    # Add a new section for name matching
    st.sidebar.header("Name Matching")
    
    # Find name and email columns
    name_column = None
    email_column = None
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            # Check if column might contain names
            if df[col].astype(str).str.replace(' ', '').str.isalpha().all():
                name_column = col
            # Check if column might contain emails
            elif df[col].astype(str).str.contains('@').any():
                email_column = col
    
    if name_column:
        st.sidebar.write(f"Name column detected: {name_column}")
        if email_column:
            st.sidebar.write(f"Email column detected: {email_column}")
        
        # Add name matching controls
        if st.sidebar.button("Find Similar Names"):
            with st.spinner("Analyzing names..."):
                matches = find_similar_names(df, name_column, email_column)
                st.session_state.name_matches = matches
                
                if matches:
                    st.sidebar.success(f"Found {len(matches)} potential matches!")
                    
                    # Display matches in a table
                    st.subheader("Potential Name Matches")
                    matches_df = pd.DataFrame(matches)
                    matches_df['confidence'] = matches_df['confidence'].round(1)
                    
                    # Add merge controls
                    for idx, match in matches.iterrows():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"**{match['name1']}**")
                        with col2:
                            st.write(f"**{match['name2']}**")
                        with col3:
                            confidence = match['confidence']
                            st.write(f"Confidence: {confidence}%")
                            if confidence >= 85:
                                st.success("High confidence match")
                            elif confidence >= 70:
                                st.warning("Medium confidence match")
                            else:
                                st.error("Low confidence match")
                    
                    # Add merge button
                    if st.button("Merge High Confidence Matches"):
                        st.session_state.merged_df = merge_similar_records(df, matches, name_column, email_column)
                        st.success("Records merged successfully!")
                        
                        # Display merge summary
                        st.subheader("Merge Summary")
                        st.write(f"Original unique names: {len(df[name_column].unique())}")
                        st.write(f"Merged unique names: {len(st.session_state.merged_df[name_column].unique())}")
                        st.write(f"Names merged: {len(df[name_column].unique()) - len(st.session_state.merged_df[name_column].unique())}")
                        
                        # Add option to use merged data
                        if st.button("Use Merged Data"):
                            st.session_state.df = st.session_state.merged_df
                            st.session_state.unique_names = get_unique_names(st.session_state.df)
                            st.session_state.unique_events = get_unique_events(st.session_state.df)
                            st.success("Switched to merged data!")
                else:
                    st.sidebar.info("No similar names found.")
    else:
        st.sidebar.warning("No name column detected in the data.")
else:
    st.info("Please upload an Excel file to begin analysis") 