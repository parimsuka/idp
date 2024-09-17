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
    
        # Store the element-wise split data for this section
        sections_data[section] = element_data

    return sections_data
