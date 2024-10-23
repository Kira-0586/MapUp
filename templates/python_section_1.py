from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    lst = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        result.extend(group[::-1])
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for word in lst:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)
    
    return dict(sorted(length_dict.items())) 


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
        def permute(nums, path=[]):
        if not nums:
            result.append(path)
        else:
            seen = set()
            for i in range(len(nums)):
                if nums[i] not in seen:
                    seen.add(nums[i])
                    permute(nums[:i] + nums[i+1:], path + [nums[i]])
    pass



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Args:
        text (str): A string containing the dates in various formats.
    
    Returns:
        List[str]: A list of valid dates in the formats specified.
    """
    dates = []
    parts = text.split()  # Split the text by space
    for word in parts:
        # Check for "dd-mm-yyyy" pattern
        if len(word) == 10 and word[2] == '-' and word[5] == '-' and word[:2].isdigit() and word[3:5].isdigit() and word[6:].isdigit():
            dates.append(word)
        # Check for "mm/dd/yyyy" pattern
        elif len(word) == 10 and word[2] == '/' and word[5] == '/' and word[:2].isdigit() and word[3:5].isdigit() and word[6:].isdigit():
            dates.append(word)
        # Check for "yyyy.mm.dd" pattern
        elif len(word) == 10 and word[4] == '.' and word[7] == '.' and word[:4].isdigit() and word[5:7].isdigit() and word[8:].isdigit():
            dates.append(word)
    
    return dates
    pass


import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    
    Args:
        lat1, lon1: Latitude and Longitude of the first point.
        lat2, lon2: Latitude and Longitude of the second point.
        
    Returns:
        Distance between the two points in meters.
    """
    R = 6371000  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
       # Step 1: Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Step 2: Convert the coordinates list into a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Step 3: Calculate the distance for each row from the previous row
    distances = [0]  # First point has no previous point, so distance is 0
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude']
        lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    # Step 4: Add the distance column to the DataFrame
    df['distance'] = distances
    
    return df
    
    return pd.DataFrame()    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Rotate the matrix by 90 degrees clockwise
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Multiply each element by the sum of its original row and column
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i])
            col_sum = sum(rotated[k][j] for k in range(n))
            rotated[i][j] = row_sum + col_sum - rotated[i][j]
    
    return rotated


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Step 1: Parse start and end timestamps by combining date and time
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Step 2: Define the days of the week to track
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Step 3: Create a dictionary to track 24-hour coverage for each day of the week for each (id, id_2) pair
    coverage_dict = {}

    # Group data by 'id' and 'id_2'
    grouped = df.groupby(['id', 'id_2'])

    # Step 4: Check coverage for each group
    for (id_val, id_2_val), group in grouped:
        coverage_dict[(id_val, id_2_val)] = {}
        # Initialize each day with False indicating no coverage
        day_coverage = {day: False for day in days_of_week}
        
        # For each row in the group, check which days and times are covered
        for _, row in group.iterrows():
            start_day = row['startDay']
            end_day = row['endDay']

            # Mark the start and end day as covered if valid
            if start_day in days_of_week:
                day_coverage[start_day] = True
            if end_day in days_of_week:
                day_coverage[end_day] = True

        # Check if all days have full coverage
        all_days_covered = all(day_coverage.values())
        
        # If not all days are covered, mark this (id, id_2) pair as incomplete
        coverage_dict[(id_val, id_2_val)] = not all_days_covered

    # Step 5: Return the boolean series
    return pd.Series(coverage_dict)
