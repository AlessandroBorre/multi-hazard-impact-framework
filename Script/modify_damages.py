from impact_calculation import load_earthquake_vulnerability_curve, load_flood_vulnerability_curve, calculate_flood_impact, calculate_earthquake_impact
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.stats import lognorm

# Function to calculate damage based on a single hazard type and intensity
def f1(hazard, intensity, use_destination, building_type, floors, flood_curve_paths, earthquake_curve_paths):
    """
    Calculate damage based on the hazard type and intensity.
    - hazard: The type of the hazard ('EQ' for earthquake, 'FL' for flood).
    - intensity: The intensity of the hazard.
    - use_destination: The type of building (e.g., Residential, Commercial).
    - building_type: The type of building (e.g., Masonry, Steel).
    - floors: Number of floors of the building.
    - flood_curve_paths: Paths to the flood vulnerability curves.
    - earthquake_curve_paths: Paths to the earthquake vulnerability curves.

    Returns the calculated damage (value between 0 and 1).
    """
    # Calculate the damage based on the hazard type
    try:
        if hazard == 'FL':
            # Calculate flood impact using logarithmic interpolation
            damage = calculate_flood_impact(intensity, use_destination, flood_curve_paths)
        elif hazard == 'EQ':
            # Calculate earthquake impact using lognormal distribution
            damage = calculate_earthquake_impact(intensity, use_destination, building_type, floors, earthquake_curve_paths)
        else:
            raise ValueError(f"Unknown hazard type: {hazard}")

        return np.clip(damage, 0, 1)  # Ensure the damage is between 0 and 1

    except Exception as e:
        #print(f"Error in f1 calculation for hazard {hazard}: {e}")
        return 0  # Return 0 in case of failure


def f2(hazard1, intensity1, hazard2, intensity2, use_destination, building_type, floors, flood_curve_paths, earthquake_curve_paths):
    """
    Creates a combined vulnerability surface (EQ + FL or FL + EQ) and returns the combined damage.
    Manages the combination of hazards FL-EQ or EQ-FL.
    Optimized for speed using vectorized operations.

    Parameters:
    - hazard1: First hazard type ('EQ' for Earthquake, 'FL' for Flood)
    - intensity1: Intensity of the first hazard
    - hazard2: Second hazard type ('EQ' for Earthquake, 'FL' for Flood)
    - intensity2: Intensity of the second hazard
    - use_destination: Type of use (e.g., residential, commercial)
    - building_type: Type of building (e.g., masonry, concrete)
    - floors: Number of floors
    - flood_curve_paths: Paths to the flood vulnerability curves
    - earthquake_curve_paths: Paths to the earthquake vulnerability curves

    Returns:
    - Combined damage value between 0 and 1
    """
    try:
        # Load vulnerability curves for hazard1
        flood_interp_function1, eq_median1, eq_sigma1 = None, None, None
        if hazard1 == 'FL':
            flood_curve1 = load_flood_vulnerability_curve(use_destination, flood_curve_paths)
            flood_intensities1 = flood_curve1['Flood Depth [m]'].values
            flood_damages1 = flood_curve1['Damage'].values
            log_flood_intensities1 = np.log(np.maximum(flood_intensities1, 0.01))  # Avoid log(0)
            flood_interp_function1 = interp1d(log_flood_intensities1, flood_damages1, kind='linear', fill_value="extrapolate")
        elif hazard1 == 'EQ':
            eq_median1, eq_sigma1 = load_earthquake_vulnerability_curve(use_destination, building_type, floors, earthquake_curve_paths)

        # Load vulnerability curves for hazard2
        flood_interp_function2, eq_median2, eq_sigma2 = None, None, None
        if hazard2 == 'FL':
            flood_curve2 = load_flood_vulnerability_curve(use_destination, flood_curve_paths)
            flood_intensities2 = flood_curve2['Flood Depth [m]'].values
            flood_damages2 = flood_curve2['Damage'].values
            log_flood_intensities2 = np.log(np.maximum(flood_intensities2, 0.01))  # Avoid log(0)
            flood_interp_function2 = interp1d(log_flood_intensities2, flood_damages2, kind='linear', fill_value="extrapolate")
        elif hazard2 == 'EQ':
            eq_median2, eq_sigma2 = load_earthquake_vulnerability_curve(use_destination, building_type, floors, earthquake_curve_paths)

        # Create intensity grids
        intensity_grid_flood = np.linspace(0.1, 6.0, 50)  # Flood intensity range
        intensity_grid_eq = np.linspace(0.01, 1.0, 50)  # Earthquake intensity range
        I1, I2 = np.meshgrid(intensity_grid_flood, intensity_grid_eq)

        # Calculate damage grid for hazard1
        flood_damage_grid1 = np.zeros_like(I1)
        eq_damage_grid1 = np.zeros_like(I1)
        if hazard1 == 'FL':
            log_I1 = np.log(np.maximum(I1, 0.01))  # Avoid log(0)
            flood_damage_grid1 = flood_interp_function1(log_I1)
        elif hazard1 == 'EQ':
            eq_damage_grid1 = lognorm.cdf(np.maximum(I1, 0.01), eq_sigma1, scale=eq_median1)

        # Calculate damage grid for hazard2
        flood_damage_grid2 = np.zeros_like(I2)
        eq_damage_grid2 = np.zeros_like(I2)
        if hazard2 == 'FL':
            log_I2 = np.log(np.maximum(I2, 0.01))  # Avoid log(0)
            flood_damage_grid2 = flood_interp_function2(log_I2)
        elif hazard2 == 'EQ':
            eq_damage_grid2 = lognorm.cdf(np.maximum(I2, 0.01), eq_sigma2, scale=eq_median2)

        # Combine the vulnerability grids using vectorized operations
        combined_damage_grid = np.clip(flood_damage_grid1 + eq_damage_grid1 + flood_damage_grid2 + eq_damage_grid2, 0, 1)

        # Calculate the maximum individual damage for each grid cell
        max_individual_damage = np.maximum.reduce([flood_damage_grid1, eq_damage_grid1, flood_damage_grid2, eq_damage_grid2])

        # Ensure the combined damage is at least the maximum of individual damages
        combined_damage_grid = np.maximum(combined_damage_grid, max_individual_damage)

        # Clip intensities to avoid out-of-bounds errors
        intensity1 = np.clip(intensity1, intensity_grid_flood.min(), intensity_grid_flood.max())
        intensity2 = np.clip(intensity2, intensity_grid_eq.min(), intensity_grid_eq.max())

        # Interpolate combined damage for the given intensities
        combined_damage = griddata((I1.ravel(), I2.ravel()), combined_damage_grid.ravel(), (intensity1, intensity2), method='linear')

        # Ensure combined damage does not exceed 1
        if combined_damage > 1:
            combined_damage = 0.99

        # Handle missing or NaN values
        if combined_damage is None or np.isnan(combined_damage):
            combined_damage = 0

        return combined_damage

    except Exception as e:
        print(f"Error in f2: {e}")
        return 0

# Function to modify the vulnerability curve of the second hazard based on the damage from the first hazard
def f3(damage1, intensity2, delta_time, hazard2, use_destination, building_type, floors, flood_curve_paths, earthquake_curve_paths):
    """
    Modifies the vulnerability curve of the second hazard based on:
    - damage1: The damage caused by the first hazard.
    - intensity2: The intensity of the second hazard.
    - delta_time: The time difference between the first and second events (ts[i+1] - te[i]).
    - hazard2: The type of the second hazard ('FL' for flood, 'EQ' for earthquake).

    The function returns the modified damage for the second hazard.
    """

    # Calculate the original damage for the second hazard using the existing vulnerability functions
    if hazard2 == 'FL':
        original_damage2 = calculate_flood_impact(intensity2, use_destination, flood_curve_paths)
    elif hazard2 == 'EQ':
        original_damage2 = calculate_earthquake_impact(intensity2, use_destination, building_type, floors, earthquake_curve_paths)
    else:
        raise ValueError(f"Unknown hazard type: {hazard2}")

    # Define a state-dependent modification factor for the second hazard
    # Shorter delta_time means less recovery, amplifying the damage

    # Adjust recovery factor: using a more gradual decay
    recovery_factor = np.log(1 + delta_time / 30)  # Logarithmic recovery factor slows down the decay effect

    # Scale the amplification by using a smoother function of `damage1`
    # Reduce how strongly `damage1` impacts the second hazard
    damage_amplification = 1 + (damage1 ** 0.8) * recovery_factor  # Use exponent < 1 to reduce rapid growth of damage

    # Apply this moderated amplification to the second hazard's damage
    modified_damage2 = original_damage2 * damage_amplification

    # Ensure damage remains between 0 and 1 but avoid harsh clipping
    if modified_damage2 > 1:
        modified_damage2 = 1 - (1 - original_damage2) * (1 - damage1 * 0.2)  # Moderate the overflow if damage goes above 1

    return modified_damage2


# Note:
# The functions f1, f2, and f3 calculate initial and adjusted damages based on intensity values.
# These functions are used to recalculate damages when events are consecutive in time.
# To use other criteria for calculating damages, modify the logic within these functions.
