import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_event_generation import generate_random_events
from csv_event_loader import load_events_from_csv
from exposure_reader import load_exposure
from recovery_functions import recovery_velocity_shape, funcResp, funcRec, R_linear, funcRecExp, funcRecLog
from impact_calculation import calculate_building_event_impacts
from modify_damages import f1, f2, f3

# Paths to the vulnerability curve files
flood_curve_paths = {
    'RES': '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/FL/FL_RES_VulnCurve.csv',
    'COM': '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/FL/FL_COM_VulnCurve.csv',
    'IND': '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/FL/FL_IND_VulnCurve.csv'
}
earthquake_curve_paths = {
    'RES': '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/EQ/EQ_RES_VulnCurve.csv',
    'COM': '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/EQ/EQ_COM_VulnCurve.csv',
    'IND': '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/EQ/EQ_IND_VulnCurve.csv'
}

# Load the vulnerability curves
use_random = input("Do you want to use random event generation? (yes/no): ").strip().lower()

if use_random == 'yes':
    num_events = int(input("Enter the number of events: "))
    events_df = generate_random_events(num_events)
else:
    # Specify the path to the CSV file directly
    file_path = '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/events.csv'
    events_df = load_events_from_csv(file_path)
print(events_df)

# Load exposure data
exposure_file_path = '/Users/alessandroborre/Library/CloudStorage/OneDrive-CIMAFoundation/Dottorato/Paper Framework/Python Code/Exposure-Table.csv'
exposure_df = load_exposure(exposure_file_path)

# Calculate impacts using the external function
impact_df = calculate_building_event_impacts(exposure_df, events_df, flood_curve_paths, earthquake_curve_paths)

# Merge the impact results into events_df or further processing
events_df = events_df.merge(impact_df[['Building ID', 'Event ID', 'Impact']], left_on=['Index', 'Index'], right_on=['Building ID', 'Event ID'], how='left')

# Ensure damages are capped at 1
events_df['Impact'] = events_df['Impact'].clip(upper=1)

# Get recovery parameters (select once for all events)
response_time, recovery_time, curve_type = recovery_velocity_shape()
print(f"Selected recovery parameters - Response Time: {response_time}, Recovery Time: {recovery_time}, Curve Type: {curve_type}")

# Get unique Building IDs from impact_df
unique_building_ids = impact_df['Building ID'].unique()

# Iterate over each Building ID
for building_id in unique_building_ids:
    # Filter the impact_df to only include rows for the current Building ID
    building_events_df = impact_df[impact_df['Building ID'] == building_id].copy()
    exposure_events_df = exposure_df[exposure_df['Index'] == building_id].copy()

    # Check if 'Building Type' is present
    if 'Building Type' not in exposure_events_df.columns:
        raise ValueError("Column 'Building Type' is missing after filtering!")
    building_type = exposure_events_df['Building Type'].iloc[0]

    # Check if the curves are available for the use destination
    use_destination = exposure_events_df['Use Destination'].iloc[0]

    if use_destination not in earthquake_curve_paths:
        print(f"No earthquake curve found for use destination: {use_destination}")
        continue  # Skip this building if there is no earthquake curve

    if use_destination not in flood_curve_paths:
        print(f"No flood curve found for use destination: {use_destination}")
        continue  # Skip this building if there is no flood curve

    # Convert 'Start Time' and 'End Time' to datetime if necessary
    if not np.issubdtype(building_events_df['Start Time'].dtype, np.datetime64):
        building_events_df['Start Time'] = pd.to_datetime(building_events_df['Start Time'])
    if not np.issubdtype(building_events_df['End Time'].dtype, np.datetime64):
        building_events_df['End Time'] = pd.to_datetime(building_events_df['End Time'])

    # Calculate the number of events and number of days based on events_df
    num_events = len(building_events_df)
    num_days = (building_events_df['End Time'].max() - building_events_df['Start Time'].min()).days
    print(f"Building ID: {building_id}, Number of events: {num_events}, Number of days: {num_days}")

    # Adjust Start Time and End Time with t0 = Start Time of the first event
    t0 = building_events_df['Start Time'].min()  # Define the reference starting point (t0)

    # Convert Start Time and End Time to the number of days from t0
    building_events_df.loc[:, 'Start Time'] = (building_events_df['Start Time'] - t0).dt.days + 1  # Days from t0 (start day 1)
    building_events_df.loc[:, 'End Time'] = (building_events_df['End Time'] - t0).dt.days + 1  # Days from t0 (end day)

    # Print the adjusted Start Time and End Time in terms of days from t0
    print(f"Adjusted Start Time and End Time for Building ID {building_id}:")
    print(building_events_df[['Start Time', 'End Time']])

    # Initialize recovery parameters
    xResp1 = 0
    xResp2 = response_time
    xRec1 = 0
    xRec2 = recovery_time
    yResp = [0, 1]
    xResp = [xResp1, xResp2]
    mResp = (yResp[1] - yResp[0]) / (xResp[1] - xResp[0])
    qResp = yResp[0] - (mResp * xResp[0])
    yRec, xRec = [0, 1], [xRec1, xRec2]
    mRec = (yRec[1] - yRec[0]) / (xRec[1] - xRec[0])
    qRec = yRec[0] - (mRec * xRec[0])

    # Selection of recovery functions to be employed based on the type of recovery chosen
    if curve_type.lower() == 'lin':
        recovery_function = R_linear
    elif curve_type.lower() == 'exp':
        recovery_function = funcRecExp
    elif curve_type.lower() == 'log':
        recovery_function = funcRecLog
    else:
        raise ValueError(f"Unknown recovery equation type: {curve_type}")

    # Initialization of three vectors ts, te, and Impact from the initial DataFrame
    ts = building_events_df['Start Time'].values
    te = building_events_df['End Time'].values
    Impact = list(building_events_df['Impact'].values)
    f1start = list(building_events_df['Impact'].values)
    print(Impact)

    # Initialization of some important parameters for the continuation of the general script, which are explained each time they are introduced
    num_days = int(max(te)+365) # Let’s add a few days to avoid limiting the graphs by considering the response and recovery of the last event
    x = np.arange(num_days) # Xs for the creation of the graph
    # Initial counting in the ratio between events
    Consecutive = 0
    Independent = 0
    Contemporary = 0
    Typology = {} # Initialization of a descriptive vector of the relationship between events
    curv = curve_type # Curve typology
    # Initialization of response and recovery times for each event
    trec = np.zeros([num_events + 1, 1])
    tresp = np.zeros([num_events + 1, 1])
    d = np.zeros(num_days) # damage vector over time
    R = np.zeros([num_days, 1]) # recovery value over time
    Int = np.ones(num_days) # physical integrity over time
    Mu = 0 # parameter for the event relationship selection (based on paper)
    rim = np.zeros([num_events + 1, 1])
    u = np.zeros([num_events + 1, 1])
    a = np.zeros([num_events + 1, 1])
    function3= np.zeros([num_events + 1, 1])

    for i in range(5 - 1):
        tresp[i] = round(int(funcResp(Impact[i], mResp, qResp)))
        print(f"The Response Time for the Hazard number {i + 1} is: {tresp[i]} days")
        delta = int(ts[i + 1] - (te[i] + tresp[i, 0]))
        print(f"The delta between the end of the Hazard number {i + 1} and the start of the Hazard number {i + 2} is: {delta} days.")

        if delta <= 0:
            teta = 0
            print(f"Hazard {i + 1} and Hazard {i + 2} are Contemporary")
            Contemporary += 1
            Impact[i] = f2(building_events_df.iloc[i]['Hazard'],building_events_df.iloc[i]['Intensity'],building_events_df.iloc[i + 1]['Hazard'],building_events_df.iloc[i + 1]['Intensity'],exposure_events_df['Use Destination'].iloc[0],building_type,exposure_events_df['Floors'].iloc[0],flood_curve_paths, earthquake_curve_paths)
            Impact[i + 1] = Impact[i]

            print(f"Evento {i}: Hazard1: {building_events_df.iloc[i]['Hazard']}, Intensity1: {building_events_df.iloc[i]['Intensity']}")
            print(f"Evento {i + 1}: Hazard2: {building_events_df.iloc[i + 1]['Hazard']}, Intensity2: {building_events_df.iloc[i + 1]['Intensity']}")
            print(f"Use Destination: {exposure_events_df['Use Destination'].iloc[0]}, Building Type: {building_type}, Floors: {exposure_events_df['Floors'].iloc[0]}")

            # Chiamata a f2
            Impact[i] = f2(
                building_events_df.iloc[i]['Hazard'],
                building_events_df.iloc[i]['Intensity'],
                building_events_df.iloc[i + 1]['Hazard'],
                building_events_df.iloc[i + 1]['Intensity'],
                exposure_events_df['Use Destination'].iloc[0],
                building_type,
                exposure_events_df['Floors'].iloc[0],
                flood_curve_paths,
                earthquake_curve_paths
            )
            Impact[i + 1] = Impact[i]
            f1start[i] = Impact[i]
            f1start[i+1] = Impact[i+1]
            tresp[i, 0] = int(funcResp(Impact[i], mResp, qResp))  # Calculate the new response time of combined damage
            tresp[i + 1, 0] = int(tresp[i, 0])
            print(f"The response time for events {i + 1} and {i + 2} is: {tresp[i, 0]} days")
            trec[i, 0] = int(funcRec(Impact[i], mRec, qRec))  # Calculate the new recovery time of combined damage
            trec[i + 1, 0] = int(trec[i, 0])
            print(f"The recovery time for events {i + 1} and {i + 2} is: {trec[i, 0]} days")
            ts[i + 1] = ts[i]  # Change the starting of the combined event
            te[i] = te[i+1]
            print(f"The damage is: {Impact[i + 1]}")
            Typology[i] = 'Contemporary'

            for j in range(int(te[i + 1] + tresp[i + 1, 0]), int(te[i + 1] + tresp[i + 1, 0] + int(trec[i + 1, 0]))):
                R[j, 0] = recovery_function(j, Impact[i + 1], int(te[i + 1] + tresp[i + 1, 0].item()),int(te[i + 1] + tresp[i + 1, 0].item() + int(trec[i + 1, 0].item())))

        else:
            teta = 1
            tresp[i, 0] = int(funcResp(Impact[i], mResp, qResp))  # Calculate the response time of the first event
            print(f"The response time for event {i + 1} is: {tresp[i, 0]} days")
            trec[i, 0] = int(funcRec(Impact[i], mRec, qRec))  # Calculate the recovery time of the first event
            print(f"The recovery time for event {i + 1} is: {trec[i, 0]} days")

            if int(ts[i + 1]) < (int(ts[i]) + int(trec[i, 0]) + int(tresp[i, 0])):
                Mu = 0
                Consecutive += 1
                print('Consecutive')
                for j in range(int(te[i] + tresp[i, 0]),int(ts[i+1])):
                    R[j, 0] = recovery_function(j, Impact[i], int(te[i] + tresp[i, 0]),int(te[i] + tresp[i, 0]) + int(trec[i,0]))
                f=R[int(ts[i + 1]), 0].item()
                rim[i+1] = (Impact[i] * R[int(ts[i + 1])-1, 0].item())
                aa= (Impact[i] * R[int(ts[i + 1])-1, 0].item())
                u[i]= round(float(f3(Impact[i], building_events_df.iloc[i + 1]['Intensity'], ts[i + 1] - te[i], building_events_df.iloc[i + 1]['Hazard'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths).item())*(1-f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths)*R[ts[i + 1]].item()) + rim[i+1].item(), 4)
                Impact[i + 1] = round(float(f3(Impact[i], building_events_df.iloc[i + 1]['Intensity'], ts[i + 1] - te[i], building_events_df.iloc[i + 1]['Hazard'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths).item()) + rim[i+1].item(), 2)
                function3[i+1]=Impact[i+1]
                if Impact[i + 1] >= 1:
                    Impact[i + 1]=0.99
                    function3[i + 1]=0.99
                k=(1-f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths)*R[int(ts[i + 1])-1, 0].item())
                h=rim[i+1].item()
                Impact[i + 1] = round(float(function3[i + 1, 0] * (1 - f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4)
                f1start[i + 1] = round(float(function3[i + 1, 0] * (1 - f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4)
                print(f"The new normalized damage value is: {u[i]}")
                tresp[i + 1, 0] = round(int(funcResp((Impact[i + 1]), mResp, qResp)))
                print(f"The new response time for the normalized damage is: {tresp[i + 1, 0]} days")
                trec[i + 1, 0] = round(int(funcRec((Impact[i + 1]), mRec, qRec)))
                print(f"The new long-term recovery time for the normalized damage is: {trec[i + 1, 0]} days")
                for j in range(int(te[i] + tresp[i, 0]),int(ts[i+1])):
                    R[j, 0] = recovery_function(j, Impact[i], int(te[i] + tresp[i, 0]),int(te[i] + tresp[i, 0]) + int(trec[i,0]))
                Typology[i] = 'Consecutive'
            else:
                Mu = 1
                Independent += 1
                Typology[i] = 'Independent'

                for j in range(int(te[i] + tresp[i, 0]), int(te[i] + tresp[i, 0] + int(trec[i, 0]))):
                    R[j, 0] = recovery_function(j, Impact[i], int(te[i] + tresp[i, 0]),int(te[i] + tresp[i, 0] + int(trec[i, 0])))
                for j in range(int(te[i + 1] + tresp[i + 1, 0]),int(te[i + 1] + tresp[i + 1, 0] + int(trec[i + 1, 0]))):
                    R[j, 0] = recovery_function(j, Impact[i + 1], int(te[i + 1] + tresp[i + 1, 0]),int(te[i + 1] + tresp[i + 1, 0] + int(trec[i + 1, 0])))


        for t in range(int(ts[i]), teta * int(te[i + 1] + tresp[i + 1, 0] + trec[i + 1, 0]) + (1 - teta) * int(te[i + 1] + tresp[i + 1, 0] + trec[i + 1, 0])):
            if int(ts[i]) <= t < teta * int(te[i] + tresp[i, 0]) + (1 - teta) * ts[i + 1]:
                d[t] = teta * f1start[i] + (1 - teta) * f1start[i]
                #d[t] = teta * f1start[i] + (1 - teta) * f2(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], building_events_df.iloc[i + 1]['Hazard'], building_events_df.iloc[i + 1]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths)
                #d[t] = teta * f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) + (1 - teta) * f2(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], building_events_df.iloc[i + 1]['Hazard'], building_events_df.iloc[i + 1]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], earthquake_curve_paths, flood_curve_paths)
                Int[t] = 1 - d[t]
            elif int(te[i] + tresp[i, 0]) <= t < int(ts[i + 1]) and teta == 1:
                d[t] = teta * f1start[i] * R[t].item()
                #d[t] = teta * f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) * R[t].item()
                Int[t] = 1 - d[t]
            elif int(ts[i + 1]) <= t < teta * int(te[i + 1] + tresp[i + 1, 0]) + (1 - teta) * int(te[i + 1] + tresp[i + 1, 0]):
                d[t] = teta * (Mu * f1start[i] + (1 - Mu) * (round(float(function3[i + 1, 0] * (1 - f1start[i] * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4))) + (1 - teta) * f1start[i]
                #d[t] = teta * (Mu * f1start[i] + (1 - Mu) * (round(float(function3[i + 1, 0] * (1 - f1start[i] * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4))) + (1 - teta) * f2(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], building_events_df.iloc[i + 1]['Hazard'], building_events_df.iloc[i + 1]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths)
                #d[t] = teta * (Mu * f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) + (1 - Mu) * (round(float(function3[i + 1, 0] * (1 - f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4))) + (1 - teta) * f2(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], building_events_df.iloc[i + 1]['Hazard'], building_events_df.iloc[i + 1]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], earthquake_curve_paths, flood_curve_paths) * R[t].item()
                Int[t] = 1 - d[t]
            elif teta * int(te[i + 1] + tresp[i + 1, 0]) + (1 - teta) * int(te[i + 1] + tresp[i + 1, 0]) <= t < teta * int(te[i + 1] + tresp[i + 1, 0] + trec[i + 1, 0]) + (1 - teta) * int(te[i + 1] + tresp[i + 1, 0] + trec[i + 1, 0]):
                d[t] = teta * (Mu * f1start[i] + (1 - Mu) * (round(float(function3[i + 1, 0] * (1 - f1start[i] * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4))*R[t].item()) + (1 - teta) * f2(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], building_events_df.iloc[i + 1]['Hazard'], building_events_df.iloc[i + 1]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) * R[t].item()
                #d[t] = teta * (Mu * f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) + (1 - Mu) * (round(float(function3[i + 1, 0] * (1 - f1(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], flood_curve_paths, earthquake_curve_paths) * R[int(ts[i + 1]) - 1, 0]) + rim[i + 1, 0]), 4))*R[t].item()) + (1 - teta) * f2(building_events_df.iloc[i]['Hazard'], building_events_df.iloc[i]['Intensity'], building_events_df.iloc[i + 1]['Hazard'], building_events_df.iloc[i + 1]['Intensity'], exposure_events_df['Use Destination'].iloc[0], building_type, exposure_events_df['Floors'].iloc[0], earthquake_curve_paths, flood_curve_paths) * R[t].item()
                Int[t] = 1 - d[t]
            else:
                d[t] = 0
                Int[t] = 1 - d[t]
        a=1

    analysisT = te[i] + tresp[i] + trec[i] + 50
    # Plot of integrity over time
    plt.close()
    plt.plot(x[0:int(analysisT[0])], Int[0:int(analysisT[0])], label='Integrity')
    plt.xlabel('Days')
    plt.ylabel('Integrity')
    plt.title('Integrity over time')
    plt.xlim(-40, int(analysisT[0]))
    plt.legend()
    #plt.show()
    plt.savefig(f'/Users/alessandroborre/Desktop/Integ_CasoProva_{curv}.png')





