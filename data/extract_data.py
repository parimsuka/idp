import pandas as pd

def extract_sections_by_elements():
    """
    Extract sections by element horizontally for a cleaned dataset.
    
    Returns:
        dict: A dictionary with elements as top-level keys and sections as sub-keys.
    """

    # Load the Excel file
    file_path = 'data/Simualtion overview.xlsx'
    
    # Load the data, skipping empty rows and any non-relevant rows at the beginning
    data = pd.read_excel(file_path, header=1)
    
    # Clean up the data by removing fully empty rows
    data_cleaned = data#.dropna(how='all')
    
    # Define the section labels to split the data vertically
    sections = ['SingleF', 'PB', 'Climbing', 'BBD', 'Verification']

    # Find the indices where each section starts
    section_indices = data_cleaned[data_cleaned.iloc[:, 0].isin(sections)].index.tolist()
    
    # Define the starting column indices for each element (for horizontal splitting)
    element_indices = [2, 11, 21, 29]
    element_names = ['Paracetamol', 'EC', 'Kollicoat', 'PEG8000']
    
    # Initialize an empty dictionary to store each section's data
    sections_data = {}
    
    # Extract each section
    for i, section in enumerate(sections):
        start_idx = section_indices[i] + 1  # Start after the section label row
        end_idx = section_indices[i + 1] if i + 1 < len(section_indices) else None  # Until the next section or end of data
    
        # Slice the data for this section
        section_data = data_cleaned.iloc[start_idx:end_idx]
        
        # Initialize a dictionary to store horizontally split data for this section
        element_data = {}
    
        # Extract each element horizontally
        for j, elem_start in enumerate(element_indices):
            # elem_end = element_indices[j + 1] if j + 1 < len(element_indices) else None  # Until the next element or end
            element_name = element_names[j]
    
            elem_end = element_indices[j + 1] if j + 1 < len(element_indices) else section_data.shape[1] + 1  # Use the total number of columns if end is None
    
            # Slice the data for the current element
            element_df = section_data.iloc[:, elem_start-1:elem_end-1]
    
            # Store the data in the dictionary
            element_data[element_name] = element_df
            element_df.columns = [col.split('.')[0].split(':')[0] for col in element_df.columns]
    
        # Store the element-wise split data for this section
        sections_data[section] = element_data

    return sections_data

def combine_elements_data(elements_data):
    """
    Combines the extracted element data for all sections and elements.

    Args:
        elements_data (dict): Dictionary of elements and sections data.

    Returns:
        pd.DataFrame: Combined DataFrame with data from all sections and elements.
    """

    # paracetamol_singlef_data = elements_data['SingleF']['Paracetamol'].iloc[:, :-1].dropna(how='all')
    # paracetamol_pb_data = elements_data['PB']['Paracetamol'].iloc[:, :-1].dropna(how='all')
    # paracetamol_climbing_data = elements_data['Climbing']['Paracetamol'].iloc[:, :-1].dropna(how='all')
    # paracetamol_bbd_data = elements_data['BBD']['Paracetamol'].iloc[:, :-1].dropna(how='all')
    # paracetamol_verification_data = elements_data['Verification']['Paracetamol'].iloc[:, :-1].dropna(how='all')

    # ec_singlef_data = elements_data['SingleF']['EC'].iloc[:, :-2].dropna(how='all')
    # ec_pb_data = elements_data['PB']['EC'].iloc[:, :-2].dropna(how='all')
    # ec_climbing_data = elements_data['Climbing']['EC'].iloc[:, :-2].dropna(how='all')
    # ec_bbd_data = elements_data['BBD']['EC'].iloc[:, :-2].dropna(how='all')
    # ec_verification_data = elements_data['Verification']['EC'].iloc[:, :-2].iloc[:-1, :].dropna(how='all')

    # kollicoat_singlef_data = elements_data['SingleF']['Kollicoat'].dropna(how='all')
    # kollicoat_pb_data = elements_data['PB']['Kollicoat'].dropna(how='all')
    # kollicoat_climbing_data = elements_data['Climbing']['Kollicoat'].dropna(how='all')
    # kollicoat_bbd_data = elements_data['BBD']['Kollicoat'].dropna(how='all')
    # kollicoat_verification_data = elements_data['Verification']['Kollicoat'].iloc[:-1, :].dropna(how='all')

    # peg8000_singlef_data = elements_data['SingleF']['PEG8000'].dropna(how='all')
    # peg8000_pb_data = elements_data['PB']['PEG8000'].dropna(how='all')
    # peg8000_climbing_data = elements_data['Climbing']['PEG8000'].dropna(how='all')
    # peg8000_bbd_data = elements_data['BBD']['PEG8000'].dropna(how='all')
    # peg8000_verification_data = elements_data['Verification']['PEG8000'].iloc[:-1, :].dropna(how='all')

    # combined_data_list = [paracetamol_singlef_data, paracetamol_pb_data, paracetamol_climbing_data, paracetamol_bbd_data, paracetamol_verification_data,
    #                      ec_singlef_data, ec_pb_data, ec_climbing_data, ec_bbd_data, ec_verification_data,
    #                      kollicoat_singlef_data, kollicoat_pb_data, kollicoat_climbing_data, kollicoat_bbd_data, kollicoat_verification_data,
    #                      peg8000_singlef_data, peg8000_pb_data, peg8000_climbing_data, peg8000_bbd_data, peg8000_verification_data]
    # combined_data = pd.concat(combined_data_list, ignore_index=True)


    # Define the section and element names to iterate over
    sections = ['SingleF', 'PB', 'Climbing', 'BBD', 'Verification']
    element_names = ['Paracetamol', 'EC', 'Kollicoat', 'PEG8000']
    
    # Initialize a list to store the combined DataFrames
    combined_data_list = []

    # Loop through each element and section to process the data
    for element in element_names:
        for section in sections:
            # Get the data for this specific element and section
            section_data = elements_data[section][element]
            
            # Apply general cleaning steps (dropping last column or row based on element/section)
            if element == 'Paracetamol':
                section_data_cleaned = section_data.iloc[:, :-1].dropna(how='all')
            elif element == 'EC':
                section_data_cleaned = section_data.iloc[:, :-2].dropna(how='all')
            else:
                section_data_cleaned = section_data.dropna(how='all')
            
            # Additional specific cleaning for the Verification section if needed
            if section == 'Verification' and element != 'Paracetamol':
                section_data_cleaned = section_data_cleaned.iloc[:-1, :]
            
            # Add cleaned data to the combined list
            combined_data_list.append(section_data_cleaned)

    # Concatenate all DataFrames into one
    combined_data = pd.concat(combined_data_list, ignore_index=True)
    
    return combined_data
