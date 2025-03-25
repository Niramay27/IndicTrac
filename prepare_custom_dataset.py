import argparse
import json
import glob
import uuid
import os
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger("prepare_custom_dataset")

def process_file(file_path: str, source_lang: str, target_lang: str, sampling_rate: int) -> list:
    """
    Process a single JSONL file and return a list of manifest samples.

    Args:
        file_path (str): Path to the JSONL file to process.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        sampling_rate (int): Audio sampling rate.

    Returns:
        list: List of manifest samples extracted from the file.
    """
    samples = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    sample_id = str(uuid.uuid4())
                    manifest_sample = {
                        "source": {
                            "id": sample_id,
                            "text": raw_sample.get("sentence", ""),
                            "lang": source_lang,
                            "audio_local_path": raw_sample["audio"]["path"],
                            "sampling_rate": sampling_rate
                        },
                        "target": {
                            "id": sample_id,
                            "text": raw_sample["translation"],
                            "lang": target_lang,
                        }
                    }
                    samples.append(manifest_sample)
                except Exception as e:
                    logger.error("Error processing line in file %s: %s", file_path, e)
    except Exception as e:
        logger.error("Error reading file %s: %s", file_path, e)
    return samples

def prepare_manifest(input_folder: str, output_manifest: str, source_lang: str, target_lang: str, sampling_rate: int) -> None:
    """
    Reads custom JSONL files from the input folder and writes a manifest file
    in the expected structure for fine-tuning, using parallel processing.

    Args:
        input_folder (str): Directory containing the JSONL files.
        output_manifest (str): Path to the output manifest JSONL file.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        sampling_rate (int): Audio sampling rate.
    """
    pattern = os.path.join(input_folder, "combined_transcripts_audio_chunks_*.jsonl")
    file_paths = glob.glob(pattern)
    if not file_paths:
        logger.error("No JSONL files found in folder: %s", input_folder)
        return

    sample_count = 0
    with open(output_manifest, "w") as fp_out:
        # Use ProcessPoolExecutor to parallelize file processing across CPU cores
        with ProcessPoolExecutor() as executor:
            # Submit all file processing tasks
            futures = [executor.submit(process_file, file_path, source_lang, target_lang, sampling_rate) 
                       for file_path in file_paths]
            # Collect results and write to output file
            for future, file_path in zip(futures, file_paths):
                try:
                    samples = future.result()
                    for sample in samples:
                        fp_out.write(json.dumps(sample) + "\n")
                        sample_count += 1
                    logger.info("Processed file: %s", file_path)
                except Exception as e:
                    logger.error("Error processing file %s: %s", file_path, e)
    logger.info("Saved %d samples to manifest: %s", sample_count, output_manifest)

def main() -> None:
    """
    Main function to parse arguments and initiate manifest preparation.
    """
    parser = argparse.ArgumentParser(
        description="Prepare a manifest file for S2T fine-tuning from custom JSONL transcripts."
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Directory containing the custom JSONL transcript files."
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        required=True,
        help="Path to the output manifest JSONL file."
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="eng",
        help="Source language code (M4T langcode)."
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="eng",
        help="Target language code (M4T langcode)."
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Audio sampling rate (e.g., 16000)."
    )
    args = parser.parse_args()
    
    prepare_manifest(
        input_folder=str(args.input_folder),
        output_manifest=str(args.output_manifest),
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        sampling_rate=args.sampling_rate
    )

if __name__ == "__main__":
    main()