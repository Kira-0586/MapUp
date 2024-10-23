import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
        # Initialize an empty DataFrame for the matrix
    toll_ids = pd.unique(df[['id_start', 'id_end']].values.ravel())
    distance_matrix = pd.DataFrame(0, index=toll_ids, columns=toll_ids)

    # Iterate over each row and fill in the distances
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance  # Ensure symmetry
    
    # Use cumulative sum of distances if direct distance isn't available
    for i in toll_ids:
        for j in toll_ids:
            if i != j and distance_matrix.at[i, j] == 0:
                # Try finding an intermediate path
                for k in toll_ids:
                    if distance_matrix.at[i, k] != 0 and distance_matrix.at[k, j] != 0:
                        distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                        distance_matrix.at[j, i] = distance_matrix.at[i, j]
    
    return distance_matrix
    distance_matrix = calculate_distance_matrix(df)

    distance_matrix.head()


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
        unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])

    return pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df.head()) 


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows where id_start is the reference_id
    ref_df = df[df['id_start'] == reference_id]
    
    # Calculate average distance
    avg_distance = ref_df['distance'].mean()
    # Set threshold as 10% of the average distance
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1
    
    # Find ids that have distances within the threshold
    ids_within_threshold = ref_df[(ref_df['distance'] >= lower_bound) & 
                                  (ref_df['distance'] <= upper_bound)]['id_end'].tolist()
    
    return sorted(ids_within_threshold)

result = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1)

    print(result)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
        # Define the rate coefficients
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Create columns for each vehicle type based on the distance
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df
    
toll_rates_df = calculate_toll_rate(unrolled_df)

    print(toll_rates_df.head())


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    from datetime import time

def calculate_time_based_toll_rates(df):
    # Create an empty list to store the new rows
    new_rows = []

    # Define the time periods for the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    # Define time intervals and discount factors
    time_intervals = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),  # 00:00:00 to 10:00:00 -> 0.8 discount
        (time(10, 0, 0), time(18, 0, 0), 1.2),  # 10:00:00 to 18:00:00 -> 1.2 multiplier
        (time(18, 0, 0), time(23, 59, 59), 0.8)  # 18:00:00 to 23:59:59 -> 0.8 discount
    ]
    
    # Loop through each unique (id_start, id_end) pair
    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        for day in weekdays + weekends:
            for start_time, end_time, factor in time_intervals:
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    # Apply the discount factor to each vehicle type
                    'moto': row['moto'] * factor,
                    'car': row['car'] * factor,
                    'rv': row['rv'] * factor,
                    'bus': row['bus'] * factor,
                    'truck': row['truck'] * factor
                }
                # Special case for weekends (constant 0.7 discount)
                if day in weekends:
                    new_row['moto'] = row['moto'] * 0.7
                    new_row['car'] = row['car'] * 0.7
                    new_row['rv'] = row['rv'] * 0.7
                    new_row['bus'] = row['bus'] * 0.7
                    new_row['truck'] = row['truck'] * 0.7
                
                new_rows.append(new_row)
    
    # Create a new DataFrame from the calculated rows
    new_df = pd.DataFrame(new_rows)
    
    return new_df

time_based_rates_df = calculate_time_based_toll_rates(toll_rates_df)

print(time_based_rates_df.head())
