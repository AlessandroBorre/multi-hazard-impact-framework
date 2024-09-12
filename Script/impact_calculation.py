from scipy.interpolate import interp1d
from scipy.stats import lognorm
import numpy as np
import pandas as pd

# Paths to the vulnerability curve files
flood_curve_paths = {
    'RES': './Data/FL/FL_RES_VulnCurve.csv',
    'COM': './Data/FL/FL_COM_VulnCurve.csv',
    'IND': './Data/FL/FL_IND_VulnCurve.csv'
}

earthquake_curve_paths = {
    'RES': './Data/EQ/EQ_RES_VulnCurve.csv',
    'COM': './Data/EQ/EQ_COM_VulnCurve.csv',
    'IND': './Data/EQ/EQ_IND_VulnCurve.csv'
}

# Function to load flood vulnerability curve
def load_flood_vulnerability_curve(use_destination, flood_curve_paths):
    curve_path = flood_curve_paths.get(use_destination)
    if curve_path is None:
        raise ValueError(f"No flood curve found for use destination: {use_destination}")
    #print(f"Loading flood curve from: {curve_path}")  # Add debug statement
    curve_data = pd.read_csv(curve_path)
    return curve_data

def load_earthquake_vulnerability_curve(use_destination, building_type, floors, earthquake_curve_paths):
    curve_path = earthquake_curve_paths.get(use_destination)

    if curve_path is None:
        raise ValueError(f"No earthquake curve found for use destination: {use_destination}")

    # Caricamento del file CSV
    curve_data = pd.read_csv(curve_path)

    # Verifica se le colonne necessarie esistono
    if 'Median (S_a)' not in curve_data.columns or 'Standard Deviation' not in curve_data.columns:
        raise KeyError(f"Le colonne 'Median (S_a)' o 'Standard Deviation' mancano nel file {curve_path}")

    # Verifica se la colonna 'Building Type' esiste
    if 'Building Type' not in curve_data.columns:
        print(f"Colonna 'Building Type' mancante per il file {curve_path}, ignorando il filtro per il tipo di edificio.")
        # Restituiamo il primo valore disponibile (o un'alternativa sensata)
        median = curve_data['Median (S_a)'].iloc[0]
        sigma = curve_data['Standard Deviation'].iloc[0]
        return median, sigma

    # Filtriamo la curva per Building Type e Floors
    curve_row = curve_data[(curve_data['Building Type'] == building_type) & (curve_data['Floors'] == floors)]

    if curve_row.empty:
        print(f"Nessuna curva trovata per Building Type: {building_type}, Floors: {floors}, usando valori predefiniti.")
        # Se non trova una corrispondenza, ritorniamo il primo valore disponibile
        median = curve_data['Median (S_a)'].iloc[0]
        sigma = curve_data['Standard Deviation'].iloc[0]
        return median, sigma

    # Estrazione dei valori medi e sigma
    median = curve_row['Median (S_a)'].values[0]
    sigma = curve_row['Standard Deviation'].values[0]
    return median, sigma


# Flood impact calculation with logarithmic interpolation
def calculate_flood_impact(intensity, use_destination, flood_curve_paths):
    # Load the flood vulnerability curve
    curve_data = load_flood_vulnerability_curve(use_destination, flood_curve_paths)

    # Use the correct column names
    flood_depths = curve_data['Flood Depth [m]'].values
    damages = curve_data['Damage'].values

    # Add a small offset to avoid issues with log(0)
    epsilon = 0.01  # Small value to avoid log(0)
    flood_depths = np.where(flood_depths == 0, epsilon, flood_depths)

    # Perform logarithmic interpolation
    log_flood_depths = np.log(flood_depths)

    # Interpolation function based on log(flood_depths)
    interp_function = interp1d(log_flood_depths, damages, kind='linear', fill_value="extrapolate")

    # Calculate the log of the intensity (flood depth)
    log_intensity = np.log(np.maximum(intensity, epsilon))  # Avoid log(0) for the intensity

    # Interpolate the damage for the given intensity
    damage = interp_function(log_intensity)

    # Ensure that the damage value is between 0 and 1
    return np.clip(damage, 0, 1)


# Earthquake impact calculation using lognormal distribution
def calculate_earthquake_impact(intensity, use_destination, building_type, floors, earthquake_curve_paths):
    # Load the vulnerability curve for earthquake based on Use Destination
    median, sigma = load_earthquake_vulnerability_curve(use_destination, building_type, floors, earthquake_curve_paths)

    # Add a small offset (epsilon) to avoid issues with very small intensities
    epsilon = 0.01  # Small value to avoid log(0) or very small intensities
    adjusted_intensity = np.maximum(intensity, epsilon)  # Ensure intensity is not zero

    # Calculate lognormal damage based on the adjusted intensity
    damage = lognorm.cdf(adjusted_intensity, sigma, scale=median)

    # Clip the damage value to ensure it stays between 0 and 1
    return np.clip(damage, 0, 1)


# Function to calculate impact for each building and event
def calculate_building_event_impacts(exposure_df, events_df, flood_curve_paths, earthquake_curve_paths):
    impact_results = []

    for _, exposure_row in exposure_df.iterrows():
        building_id = exposure_row['Index']
        use_destination = exposure_row['Use Destination']
        building_type = exposure_row['Building Type']
        floors = exposure_row['Floors']

        for _, event_row in events_df.iterrows():
            event_id = event_row['Index']
            hazard = event_row['Hazard']
            intensity = event_row['Intensity']

            print(f"\nProcessing event: {event_id} with hazard: {hazard}")

            if hazard == 'FL':
                #print(f"Loading flood curve for building ID {building_id}")
                impact = calculate_flood_impact(intensity, use_destination, flood_curve_paths)
            elif hazard == 'EQ':
                #print(f"Loading earthquake curve for building ID {building_id}")
                impact = calculate_earthquake_impact(intensity, use_destination, building_type, floors,
                                                     earthquake_curve_paths)
            else:
                raise ValueError(f"Unknown hazard type: {hazard}")

            # Store the result
            impact_results.append({
                'Building ID': building_id,
                'Event ID': event_id,
                'Hazard': hazard,
                'Intensity': intensity,
                'Impact': impact,
                'Start Time': event_row['Start Time'],
                'End Time': event_row['End Time']
            })

    return pd.DataFrame(impact_results)
