import os
from pheme_processor import PhemeProcessor
import warnings
warnings.filterwarnings("ignore")

def main():

    Concat_CSV = True # Set to True if you want to concatenate all CSV files into one file

    # Define the base directory where your PHEME dataset is located
    pheme_dir = r'C:\Users\matth\OneDrive\Desktop\RPDNN\data\social_context\pheme-rnr-dataset' #Pheme Directory
    
    # List of event names you want to process
    events = [
        # 'charliehebdo',
        # 'ebola-essien',
        'ferguson',
        #'germanwings-crash',
        'gurlitt',
        'ottawashooting',
        'prince-toronto'
        #'sydneysiege'
        #'putinmissing'
    ]
    
    # Create Directory "CSV_Files" in the base directory if it doesn't exist
    os.makedirs(os.path.join('CSV_Files'), exist_ok=True)

    # Optionally, specify an output directory for CSV files
    output_dir = 'CSV_Files' # Project Directory
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Instantiate the PhemeProcessor
    processor = PhemeProcessor(base_dir=pheme_dir, output_dir=output_dir)
    
    # Process all specified events
    processor.process_events(events)

    if Concat_CSV:
        # Grab CSV's from the CSV_Files directory that relate to the events specified
        csv_files = [os.path.join(output_dir, f'{event}.csv') for event in events]

        # Path for the concatenated CSV file
        all_events_csv = os.path.join(output_dir, 'all_events.csv')

        try:
            with open(all_events_csv, 'w', encoding='utf-8', newline='') as outfile:
                for i, csv_file in enumerate(csv_files):
                    if not os.path.exists(csv_file):
                        print(f"[WARN] CSV file does not exist and will be skipped: {csv_file}")
                        continue

                    with open(csv_file, 'r', encoding='utf-8') as infile:
                        # Read the header
                        header = infile.readline()
                        if i == 0:
                            # Write the header only once
                            outfile.write(header)
                        else:
                            # Skip the header for subsequent files
                            pass

                        # Write the rest of the lines
                        for line in infile:
                            outfile.write(line)
            print('All CSV files have been successfully concatenated into one file: all_events.csv')
        except Exception as e:
            print(f"[ERROR] An error occurred during CSV concatenation: {e}")

if __name__ == "__main__":
    main()
