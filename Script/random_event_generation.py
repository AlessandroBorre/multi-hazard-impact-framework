import numpy as np
import pandas as pd
from datetime import timedelta, datetime

def generate_random_events(num_events):
    # Limit the hazard types to Flood (FL) and Earthquake (EQ)
    hazards = ['EQ', 'FL']

    # Define intensity limits for Flood and Earthquake
    intensity_min_fl = 0.1
    intensity_max_fl = 5.8
    intensity_min_eq = 0.01
    intensity_max_eq = 1.0

    # Define time constraints
    min_days_between_same_type = 90  # 3 months
    max_days_between_events = 1825  # 5 years

    # Randomly assign hazard types to each event
    hazard_types = np.random.choice(hazards, num_events)

    # Generate random intensities for the specified number of events
    intensities = np.zeros(num_events)

    for i in range(num_events):
        if hazard_types[i] == 'FL':
            intensities[i] = np.random.uniform(intensity_min_fl, intensity_max_fl)
        else:
            intensities[i] = np.random.uniform(intensity_min_eq, intensity_max_eq)

    intensities = intensities.round(2)

    # Initialize arrays for start and end days
    start_days = np.zeros(num_events, dtype=int)
    end_days = np.zeros(num_events, dtype=int)

    # Set the initial date (for example, starting on 01/01/2020)
    start_date = datetime.strptime('01/01/2020', '%m/%d/%Y')

    # Generate the start and end days for the first event
    start_days[0] = np.random.randint(0, 101)
    end_days[0] = start_days[0] + np.random.randint(2, 21)

    for i in range(1, num_events):
        min_start_day = end_days[i - 1] + 1
        max_start_day = min_start_day + max_days_between_events

        while True:
            # Generate start and end days for the event
            start_days[i] = np.random.randint(min_start_day, max_start_day)
            end_days[i] = start_days[i] + np.random.randint(2, 21)

            # Check if the event is within the defined limits
            same_type_events = start_days[:i][hazard_types[:i] == hazard_types[i]]
            if len(same_type_events) > 0 and any((start_days[i] - same_type_events) < min_days_between_same_type):
                continue  # If the condition is met, regenerate start and end days
            break

    # Convert start_days and end_days to actual dates
    start_dates = [start_date + timedelta(days=int(day)) for day in start_days]
    end_dates = [start_date + timedelta(days=int(day)) for day in end_days]

    # Now, add one additional event that is 2 years after the last event
    # Ensure the last event ends 2 years (730 days) before the new event
    last_end_day = end_days[-1]
    new_start_day = int(last_end_day + 730)  # Convert to int to avoid the error
    new_end_day = int(new_start_day + np.random.randint(2, 21))

    # Randomly choose hazard type and intensity for the new event
    new_hazard_type = np.random.choice(hazards)
    if new_hazard_type == 'FL':
        new_intensity = np.random.uniform(intensity_min_fl, intensity_max_fl)
    else:
        new_intensity = np.random.uniform(intensity_min_eq, intensity_max_eq)

    # Add this new event to the arrays
    hazard_types = np.append(hazard_types, new_hazard_type)
    intensities = np.append(intensities, round(new_intensity, 2))
    start_dates.append(start_date + timedelta(days=int(new_start_day)))
    end_dates.append(start_date + timedelta(days=int(new_end_day)))

    # Create a DataFrame to store event data
    events_df = pd.DataFrame({
        'Index': np.arange(1, num_events + 2),  # +1 for the extra event
        'Hazard': hazard_types,
        'Intensity': intensities,
        'Start Time': start_dates,
        'End Time': end_dates
    })

    # Sort events by start time in ascending order
    events_df = events_df.sort_values(by='Start Time').reset_index(drop=True)

    # Convert dates to mm/dd/yyyy format
    events_df['Start Time'] = events_df['Start Time'].dt.strftime('%m/%d/%Y')
    events_df['End Time'] = events_df['End Time'].dt.strftime('%m/%d/%Y')

    return events_df

# Instructions to extend to other types of hazards:
# To add new hazard types, simply add the appropriate abbreviations to the 'hazards' list.
# For example, to add 'Tsunami' (TS), update the list as follows:
# hazards = ['EQ', 'FL', 'TS']

# Note on intensity limits:
# The intensity for Flood (FL) and Earthquake (EQ) events is set to be between 0.1 and 10.0.
# To modify these limits or add limits for other hazards, update the intensity generation code accordingly.

# Note on time constraints:
# The minimum time between events of the same type is set to 90 days (3 months).
# The maximum time between any two consecutive events is set to 1825 days (5 years).
# To modify these limits, update the respective variables in the code.

# Complete list of potential hazards with abbreviations (the list is an example and is not exhausted):
# Earthquake (EQ)
# Tsunami (TS)
# Landslide (LS)
# Flood (FL)
# Windstorm (WS)
# Wildfire (WF)

