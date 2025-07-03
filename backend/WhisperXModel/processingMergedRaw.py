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
        print(f"Ensured output directory exists: '{output_directory}' for intermediate file.")

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

    print("\n--- Starting merge and re-timestamping process for raw JSONs ---")
    start_time_merge = time.time()

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

    end_time_merge = time.time()
    total_time_merge = end_time_merge - start_time_merge
    print(f"\n--- Finished reading and offsetting {files_processed_count} raw JSON files in {total_time_merge:.2f} seconds ---")

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

def aggregate_speaker_turns(segments_data):
    """
    Aggregates consecutive text segments by the same speaker into single turns.
    Segments missing a 'speaker' key are dropped (ignored).

    Args:
        segments_data (dict): A dictionary containing a "segments" key,
                              where its value is a list of segment dictionaries.

    Returns:
        list: A list of dictionaries, each representing an aggregated speaker turn.
              Each dictionary contains 'speaker', 'text', 'start', and 'end'.
              Segments without a 'speaker' key are excluded.
    """
    segments = segments_data.get("segments", [])
    if not segments:
        print("No segments found in the input data for aggregation.")
        return []

    aggregated_turns = []
    current_turn = None # Initialize as None, will be set once the first valid segment is found

    for segment in segments:
        # Fail check: Drop segment if 'speaker' key is missing
        if "speaker" not in segment:
            print(f"Warning: Dropping segment due to missing 'speaker' key during aggregation: {segment.get('text', 'No text')}")
            continue # Skip to the next segment

        segment_speaker = segment["speaker"]
        segment_text = segment["text"].strip()
        segment_start = segment["start"]
        segment_end = segment["end"]

        if current_turn is None:
            # This is the first valid segment found, initialize current_turn
            current_turn = {
                "speaker": segment_speaker,
                "text": segment_text,
                "start": segment_start,
                "end": segment_end
            }
        elif segment_speaker == current_turn["speaker"]:
            # If the same speaker, append text and update end time
            current_turn["text"] += " " + segment_text
            current_turn["end"] = segment_end
        else:
            # If a different speaker, save the current turn and start a new one
            aggregated_turns.append(current_turn)
            current_turn = {
                "speaker": segment_speaker,
                "text": segment_text,
                "start": segment_start,
                "end": segment_end
            }
    
    # After the loop, if there's an active current_turn, add it
    if current_turn is not None:
        aggregated_turns.append(current_turn)
        
    return aggregated_turns

# --- Main Execution Flow ---
if __name__ == "__main__":
    total_pipeline_start_time = time.time()

    # --- Step 1: Merge and Re-timestamp Raw JSONs ---
    base_raw_input_path = "backend/WhisperXModel/output/raw/"
    merged_raw_output_dir = "backend/WhisperXModel/output/merged_raw/"
    intermediate_merged_raw_filename = os.path.join(merged_raw_output_dir, "full_audio_raw_transcription_with_absolute_timestamps.json")
    
    # Assuming 20-minute chunks
    CHUNK_DURATION_SECONDS = 20 * 60 

    print("\n--- Starting full transcription pipeline ---")

    merge_success = merge_and_retimestamp_raw_jsons(
        base_input_raw_dir=base_raw_input_path,
        intermediate_output_filename=intermediate_merged_raw_filename,
        chunk_duration_seconds=CHUNK_DURATION_SECONDS
    )

    if not merge_success:
        print("\nPipeline stopped: Raw JSON merging and re-timestamping failed or produced no data.")
        exit() # Exit if the first step failed

    # --- Step 2: Aggregate Speaker Turns from the Merged Raw Data ---
    final_output_dir = "backend/WhisperXModel/output/processed/"
    final_output_filename = os.path.join(final_output_dir, "outputFinal.json")

    # Ensure the final output directory exists
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Ensured final output directory exists: '{final_output_dir}' for aggregated file.")

    print("\n--- Starting speaker turn aggregation ---")
    start_time_aggregate = time.time()

    try:
        with open(intermediate_merged_raw_filename, 'r', encoding='utf-8') as f:
            podcast_segments_raw_data = json.load(f)
        
        print(f"Successfully loaded merged raw data from '{intermediate_merged_raw_filename}' for aggregation.")

        processed_aggregated_data = aggregate_speaker_turns(podcast_segments_raw_data)

        # Save the final aggregated data to a new JSON file
        with open(final_output_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_aggregated_data, f, indent=4, ensure_ascii=False)

        print(f"Aggregated speaker turns saved to '{final_output_filename}'")
        
        end_time_aggregate = time.time()
        total_time_aggregate = end_time_aggregate - start_time_aggregate
        print(f"Speaker aggregation completed in {total_time_aggregate:.2f} seconds")

        print("\n--- Aggregated Data (first few entries) ---")
        if processed_aggregated_data:
            for i, entry in enumerate(processed_aggregated_data):
                if i >= 5:
                    break
                print(f"[{entry['speaker']}] ({entry['start']:.3f} - {entry['end']:.3f}): {entry['text']}")
            if len(processed_aggregated_data) > 5:
                print("...")
        else:
            print("No final aggregated data to display.")
            
    except FileNotFoundError:
        print(f"Error: Intermediate merged file '{intermediate_merged_raw_filename}' not found for aggregation. This should not happen if the previous step was successful.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{intermediate_merged_raw_filename}' during aggregation. Please check if the file is valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred during aggregation: {e}")

    total_pipeline_end_time = time.time()
    total_pipeline_duration = total_pipeline_end_time - total_pipeline_start_time
    print(f"\n--- Total pipeline execution time: {total_pipeline_duration:.2f} seconds ---")