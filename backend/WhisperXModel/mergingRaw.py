import json
import os
import re
import time

def merge_and_retimestamp_raw_jsons(base_input_raw_dir, intermediate_output_filename, chunk_duration_seconds=1200):
    """
    Reads multiple raw WhisperX output JSON files (outputXXX.json) from subdirectories
    in base_input_raw_dir, adjusts their timestamps based on their order (assuming
    fixed-duration chunks), and merges them into a single JSON file.

    Args:
        base_input_raw_dir (str): The base directory containing 'outputXXX' subfolders
                                   with raw JSONs.
        intermediate_output_filename (str): The full path and filename for the
                                            single merged JSON output.
        chunk_duration_seconds (int): The duration of each audio chunk in seconds
                                      (default is 1200 for 20 minutes).
    Returns:
        bool: True if merging was successful and a file was created, False otherwise.
    """
    
    # Ensure the output directory for the intermediate file exists
    output_directory = os.path.dirname(intermediate_output_filename)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Ensured output directory exists: '{output_directory}'")

    all_raw_segments_merged = []
    files_processed_count = 0

    # Get a list of all items in the raw directory
    try:
        subdirs = os.listdir(base_input_raw_dir)
    except FileNotFoundError:
        print(f"Error: Base input directory '{base_input_raw_dir}' not found. Please check the path.")
        return False

    # Filter and sort folders by the numerical part (e.g., output000, output001, ...)
    # This is CRUCIAL for correct timestamp offsetting
    file_folders_to_process = sorted([
        d for d in subdirs if os.path.isdir(os.path.join(base_input_raw_dir, d)) and re.match(r'^output\d{3}$', d)
    ])

    if not file_folders_to_process:
        print(f"No 'outputXXX' folders found in '{base_input_raw_dir}'. Nothing to merge.")
        return False

    print("\n--- Starting merge and re-timestamping process ---")
    start_time = time.time()

    for idx, folder_name in enumerate(file_folders_to_process):
        input_folder_path = os.path.join(base_input_raw_dir, folder_name)
        input_file_name = f"{folder_name}.json"
        input_full_path = os.path.join(input_folder_path, input_file_name)

        current_offset = idx * chunk_duration_seconds

        print(f"Processing: {input_full_path} (Applying offset: {current_offset:.2f}s)")

        if not os.path.exists(input_full_path):
            print(f"Warning: Input file '{input_full_path}' not found. Skipping.")
            continue

        try:
            with open(input_full_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            segments_in_chunk = chunk_data.get("segments", [])
            
            # Apply offset to all timestamps in the current chunk's segments
            for segment in segments_in_chunk:
                # Adjust segment start/end
                segment['start'] += current_offset
                segment['end'] += current_offset
                
                # Adjust word start/end if words data exists
                if 'words' in segment:
                    for word in segment['words']:
                        word['start'] += current_offset
                        word['end'] += current_offset
                
                all_raw_segments_merged.append(segment)
            
            files_processed_count += 1

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{input_full_path}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while reading/offsetting '{input_full_path}': {e}. Skipping.")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    print(f"\n--- Finished reading and offsetting {files_processed_count} raw JSON files in {total_time_seconds:.2f} seconds ---")

    if not all_raw_segments_merged:
        print("No segments were collected after processing all raw files. No merged output will be created.")
        return False
    else:
        # Save the single merged raw data with adjusted timestamps
        try:
            merged_output_data = {"segments": all_raw_segments_merged}
            with open(intermediate_output_filename, 'w', encoding='utf-8') as f:
                json.dump(merged_output_data, f, indent=4, ensure_ascii=False)
            print(f"Merged raw data (with adjusted timestamps) saved to '{intermediate_output_filename}'")
            return True
        except Exception as e:
            print(f"Error saving merged raw transcription: {e}")
            return False

# --- How to use this function ---
if __name__ == "__main__":
    base_raw_input_path = "backend/WhisperXModel/output/raw/"
    merged_raw_output_path = "backend/WhisperXModel/output/merged_raw/"
    merged_raw_output_filename = os.path.join(merged_raw_output_path, "full_audio_raw_transcription_with_absolute_timestamps.json")

    # Call the function to perform the merging and re-timestamping
    success = merge_and_retimestamp_raw_jsons(
        base_input_raw_dir=base_raw_input_path,
        intermediate_output_filename=merged_raw_output_filename,
        chunk_duration_seconds=1200 # Set this to your chunk duration (e.g., 20 * 60 for 20 minutes)
    )

    if success:
        print("\n--- Merging and re-timestamping complete. ---")
        print(f"You can now use '{merged_raw_output_filename}' as input for speaker turn aggregation or other processing.")
    else:
        print("\n--- Merging and re-timestamping failed or no data to merge. ---")