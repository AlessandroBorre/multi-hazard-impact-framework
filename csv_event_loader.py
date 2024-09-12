import pandas as pd
import numpy as np

def load_events_from_csv(file_path):
    """
    Load events data from a CSV file and perform necessary checks and conversions.
    Add an extra random event that starts 2 years after the last event and ends 1 day later.

    Args:
    file_path (str): Path to the CSV file containing the events data.

    Returns:
    pd.DataFrame: DataFrame containing the events data.
    """
    # Load the events from the CSV file
    events_df = pd.read_csv(file_path)

    # Convert 'Start Time' and 'End Time' to datetime
    events_df['Start Time'] = pd.to_datetime(events_df['Start Time'], dayfirst=True)
    events_df['End Time'] = pd.to_datetime(events_df['End Time'], dayfirst=True)

    # Sort events by 'Start Time'
    events_df = events_df.sort_values(by='Start Time').reset_index(drop=True)

    # Check for minimum days between same type of events
    min_days_between_same_type = 90  # 3 months
    for i in range(len(events_df) - 1):
        event = events_df.iloc[i]
        next_event = events_df.iloc[i + 1]

        if event['Hazard'] == next_event['Hazard']:
            days_between = (next_event['Start Time'] - event['End Time']).days
            if days_between < min_days_between_same_type:
                raise ValueError(
                    f"Event {i} ({event['Hazard']}) and Event {i + 1} ({next_event['Hazard']}) are too close: {days_between} days apart.")

    # Generate an extra event 2 years after the last event
    last_end_time = events_df['End Time'].max()
    new_start_time = last_end_time + pd.DateOffset(days=730)  # Start 2 years after the last event
    new_end_time = new_start_time + pd.DateOffset(days=1)  # End 1 day after the start

    # Randomly assign hazard type and intensity
    hazards = ['EQ', 'FL']
    new_hazard = np.random.choice(hazards)

    if new_hazard == 'FL':
        new_intensity = np.random.uniform(0.1, 5.8)  # Intensity range for floods
    else:
        new_intensity = np.random.uniform(0.01, 1.0)  # Intensity range for earthquakes

    # Append the new event to the DataFrame
    new_event = pd.DataFrame({
        'Index': [len(events_df) + 1],
        'Hazard': [new_hazard],
        'Intensity': [round(new_intensity, 2)],
        'Start Time': [new_start_time],
        'End Time': [new_end_time]
    })

    events_df = pd.concat([events_df, new_event], ignore_index=True)

    return events_df


# Note:
# This function checks for events of the same hazard type that are within 90 days of each other.
# If such events are found, the function prints an error message and stops the analysis.
# The minimum time between events of the same type is set to 90 days (3 months).
# To modify this limit, update the respective variable in the code.
