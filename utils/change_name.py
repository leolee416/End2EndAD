import os

directory = os.getcwd()
print(f"Working directory: {directory}")

# Loop through all files in the directory
for filename in sorted(os.listdir(directory)):  # Sorted ensures consistent order
    if filename.endswith(".csv"):  # Only process CSV files
        old_path = os.path.join(directory, filename)
        
        # Split the filename into parts before the last underscore
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:  # Ensure the split was successful
            base_name = parts[0]  # Everything before the last underscore
            suffix = parts[1]  # The last part after the underscore
            
            # Create the new filename with '_000' inserted
            new_name = f"{base_name}_000_{suffix}"
            new_path = os.path.join(directory, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
    