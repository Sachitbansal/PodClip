import json
import os
import time

def add_words_to_aggregated_data(aggregated_json_path, raw_merged_json_path, output_json_path):
    """
    Enriches the aggregated speaker turns data with detailed word-level information
    from the raw, merged transcription data.

    Args:
        aggregated_json_path (str): Path to the JSON file containing speaker-aggregated turns
                                    (e.g., outputFinal.json).
        raw_merged_json_path (str): Path to the JSON file containing raw, time-adjusted segments
                                    with word-level details (e.g., full_audio_raw_transcription_with_absolute_timestamps.json).
        output_json_path (str): Path where the new, enriched aggregated JSON will be saved.
    Returns:
        bool: True if the process was successful, False otherwise.
    """
    print(f"\n--- Starting process to add word details to aggregated data ---")
    start_time = time.time()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: '{output_dir}'")

    # 1. Load the aggregated speaker turns data
    if not os.path.exists(aggregated_json_path):
        print(f"Error: Aggregated JSON file not found: '{aggregated_json_path}'")
        return False
    try:
        with open(aggregated_json_path, 'r', encoding='utf-8') as f:
            aggregated_turns = json.load(f)
        print(f"Successfully loaded aggregated data from '{aggregated_json_path}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{aggregated_json_path}'.")
        return False
    except Exception as e:
        print(f"An error occurred loading aggregated data: {e}")
        return False

    # 2. Load the raw, merged data with word details
    if not os.path.exists(raw_merged_json_path):
        print(f"Error: Raw merged JSON file not found: '{raw_merged_json_path}'")
        return False
    try:
        with open(raw_merged_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        raw_segments = raw_data.get("segments", [])
        print(f"Successfully loaded raw merged data from '{raw_merged_json_path}'. Found {len(raw_segments)} raw segments.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{raw_merged_json_path}'.")
        return False
    except Exception as e:
        print(f"An error occurred loading raw merged data: {e}")
        return False

    if not aggregated_turns or not raw_segments:
        print("No data to process. Either aggregated turns or raw segments are empty.")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return True

    # 3. Match and add words
    enriched_aggregated_data = []
    raw_segment_idx = 0

    for agg_turn in aggregated_turns:
        current_agg_turn_words = []
        
        # We look for raw segments that fall within the current aggregated turn's time range
        # and belong to the same speaker.
        # We continue from the last position in raw_segments_data to optimize (assuming sorted data)
        while raw_segment_idx < len(raw_segments):
            raw_segment = raw_segments[raw_segment_idx]
            
            # Check if raw segment is potentially part of the current aggregated turn
            # A raw segment must start no earlier than the aggregated turn's start
            # and end no later than the aggregated turn's end.
            # Allowing for minor overlaps due to floating point inaccuracies/WhisperX output nuances
            # We'll use a small epsilon for comparison if necessary, but direct comparison usually works.

            is_same_speaker = (raw_segment.get('speaker') == agg_turn['speaker'])
            is_within_start_bound = (raw_segment['start'] >= agg_turn['start'] - 0.001) # Small epsilon for safety
            is_within_end_bound = (raw_segment['end'] <= agg_turn['end'] + 0.001)   # Small epsilon for safety

            # If the raw segment clearly starts after the aggregated turn ends,
            # then it belongs to a future aggregated turn. Break and move to the next agg_turn.
            if raw_segment['start'] > agg_turn['end'] + 0.001:
                break 
            
            if is_same_speaker and is_within_start_bound and is_within_end_bound:
                # This raw segment belongs to the current aggregated turn
                if 'words' in raw_segment:
                    current_agg_turn_words.extend(raw_segment['words'])
                raw_segment_idx += 1 # Move to the next raw segment for the next check
            else:
                # If this raw segment belongs to a different speaker *within* the agg_turn's time,
                # or it's an edge case, we need to advance raw_segment_idx if it's too early.
                # However, the aggregation logic *should* prevent speaker changes within an agg_turn.
                # The primary reason to break is if raw_segment['start'] > agg_turn['end'].
                # If a segment is same time range but different speaker, it means our aggregation
                # was flawed or the raw data had issues for that overlap.
                
                # For robustness, if a raw segment starts within the agg_turn's bounds but has a different speaker
                # (which ideally shouldn't happen for perfectly aggregated data), we just skip it for *this* agg_turn
                # and move the raw_segment_idx forward to find the next potentially matching one.
                # More robust: If current_raw_segment['start'] < agg_turn['start'], advance raw_segment_idx
                # until it catches up.
                if raw_segment['end'] < agg_turn['start'] - 0.001: # Raw segment ended *before* current agg_turn started
                     raw_segment_idx += 1
                     continue # Keep looking for segments relevant to this agg_turn
                
                # If we're here, it means the raw_segment is either:
                # 1. Past the current agg_turn's end (handled by the break above)
                # 2. Within the agg_turn's time but a different speaker (should ideally not happen due to aggregation logic)
                # In robust scenarios, if the raw segment doesn't fit this agg_turn, it means this agg_turn is done
                # collecting words, and the raw_segment belongs to a *future* agg_turn.
                break # Move to the next aggregated turn
        
        # Add the collected words to the current aggregated turn
        agg_turn_copy = agg_turn.copy()
        agg_turn_copy['words'] = current_agg_turn_words
        enriched_aggregated_data.append(agg_turn_copy)
        
    # 4. Save the enriched data
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_aggregated_data, f, indent=4, ensure_ascii=False)
        print(f"Enriched aggregated data saved to '{output_json_path}'")
        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        print(f"Error saving enriched data: {e}")
        return False

# --- Main execution ---
if __name__ == "__main__":
    aggregated_json_file = "backend/WhisperXModel/output/processed/outputFinal.json"
    raw_merged_json_file = "backend/WhisperXModel/output/merged_raw/full_audio_raw_transcription_with_absolute_timestamps.json"
    output_enriched_json_file = "backend/WhisperXModel/output/processed/outputFinal_with_words.json" # New output file

    success = add_words_to_aggregated_data(
        aggregated_json_file,
        raw_merged_json_file,
        output_enriched_json_file
    )

    if success:
        print("\n--- Enriched output successfully generated. ---")
        # Optional: Print the first few entries of the newly created file
        try:
            with open(output_enriched_json_file, 'r', encoding='utf-8') as f:
                enriched_data_preview = json.load(f)
                print("\n--- First 2 enriched entries (preview) ---")
                for i, entry in enumerate(enriched_data_preview):
                    if i >= 2:
                        break
                    print(f"Speaker: {entry['speaker']}, Start: {entry['start']:.3f}, End: {entry['end']:.3f}")
                    print(f"Text: {entry['text']}")
                    print(f"Words Count: {len(entry['words']) if 'words' in entry else 0}")
                    if 'words' in entry and len(entry['words']) > 0:
                        print(f"First Word: {entry['words'][0]['word']} ({entry['words'][0]['start']:.3f})")
                        print(f"Last Word: {entry['words'][-1]['word']} ({entry['words'][-1]['end']:.3f})")
                    print("-" * 30)
        except Exception as e:
            print(f"Could not load or preview enriched output: {e}")
    else:
        print("\n--- Enriched output generation failed. ---")