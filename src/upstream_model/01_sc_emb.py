import os
import math
import torch
import pandas as pd
from dataset.create_all_embedding import *
from utils.constant_mapping import *

def get_file_size(file_path):
    """
    Get and print the size of a file.

    :param file_path: The path to the file.
    """
    if os.path.exists(file_path):
        # Get the size of the file in bytes
        file_size_bytes = os.path.getsize(file_path)

        # Convert the size to a more human-readable format (e.g., KB, MB, GB, etc.)
        def convert_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_name[i]}"

        # Call the function to convert the file size and print it
        file_size_readable = convert_size(file_size_bytes)
        return file_size_readable
    else:
        return "does not exist"
    
def set_device(device_index=None):
    if device_index is not None and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > device_index:
            torch.cuda.set_device(device_index)
            device = torch.device("cuda")
            print(f"Using GPU {torch.cuda.current_device()}")
            return device
        else:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
            print("Specified GPU index is out of range. Using the first GPU.")
            return device
    else:
        device = torch.device("cpu")
        print("CUDA is not available or GPU index is not specified. Using CPU.")
        return device
    
root_path = "/home/braveenan/voice_dataset"
root_speechcommand = os.path.join(root_path, "SpeechCommand")
root_voxceleb = os.path.join(root_path, "VoxCeleb")
root_iemocap = os.path.join(root_path, "IEMOCAP")

device = set_device(0)
model_option_array = ["wav2vec2_base", "wav2vec2_large", "hubert_base", "hubert_large", "wavlm_base", "wavlm_large"]
frame_pooling_array = ["mean", "std"]

for upstream_model_type in model_option_array:
    for frame_pooling_type in frame_pooling_array:
        upstreammodel_text = f"Upstream model type: {upstream_model_type}"
        print(upstreammodel_text)
        fayer_pooling_text = f"Fayer pooling type: {frame_pooling_type}"
        print(fayer_pooling_text)
        
        root_emb_path = root_path.replace("/voice_dataset", f"/embedding/{upstream_model_type}/{frame_pooling_type}")
        root_speechcommand_emb = os.path.join(root_emb_path, "SpeechCommand")
        root_voxceleb_emb = os.path.join(root_emb_path, "VoxCeleb")
        root_iemocap_emb = os.path.join(root_emb_path, "IEMOCAP")

        os.makedirs(root_speechcommand_emb, exist_ok=True)
        os.makedirs(root_voxceleb_emb, exist_ok=True)
        os.makedirs(root_iemocap_emb, exist_ok=True)

        bundle = ModelMapping.get_model_bundle(upstream_model_type)
        upstream_model = bundle.get_model()

        training_data = SPEECHCOMMANDSEmbedding (
                    root = root_speechcommand,
                    url = "speech_commands_v0.01",
                    subset = "training",
                    download=False,
                    model=upstream_model,
                    device=device,
                    frame_pooling_type=frame_pooling_type,
                )

        validating_data = SPEECHCOMMANDSEmbedding (
                    root = root_speechcommand,
                    url = "speech_commands_v0.01",
                    subset = "validation",
                    download=False,
                    model=upstream_model,
                    device=device,
                    frame_pooling_type=frame_pooling_type,
                )

        testing_data = SPEECHCOMMANDSEmbedding (
                    root = root_speechcommand,
                    url = "speech_commands_v0.01",
                    subset = "testing",
                    download=False,
                    model=upstream_model,
                    device=device,
                    frame_pooling_type=frame_pooling_type,
                )

        dataset_length_text = f"No of training data samples: {len(training_data)} \nNo of validating data samples: {len(validating_data)} \nNo of testing data samples: {len(testing_data)}"
        print(dataset_length_text)

        # Create a function to process data and write to CSV
        def process_and_write_data(data, subset, csv_file):
            for index, data_point in enumerate(data):
                (embedding, emblength, wavlength, label, wavpath) = data_point
                duration = wavlength // 16000
                embpath = os.path.join(root_speechcommand_emb, wavpath)
                embpath = embpath.replace(".wav", f"_{upstream_model_type}_{frame_pooling_type}.pt")
                embdir = embpath.replace(f"/{embpath.split('/')[-1]}", "")
                os.makedirs(embdir, exist_ok=True)
                torch.save(embedding, embpath)
                file_size_readable = get_file_size(embpath)

                # Write the data directly to the CSV file
                csv_file.write(f"{wavpath},{label},{wavlength},{duration},{emblength},{subset},{file_size_readable}\n")

                print(f"{subset} {index+1} {wavpath}")

        # Create and open the CSV file for writing
        output_csv_file = os.path.join(root_speechcommand_emb, f"speechcommand_{upstream_model_type}_{frame_pooling_type}.csv")
        with open(output_csv_file, 'w') as csv_file:
            csv_file.write("Audio path,Audio label,Audio length,Audio duration,Sequence length,Data subset, File size\n")

            process_and_write_data(training_data, "train", csv_file)
            process_and_write_data(validating_data, "validation", csv_file)
            process_and_write_data(testing_data, "test", csv_file)
            
        # Display GPU memory usage before and after emptying the cache
        allocated_before = torch.cuda.memory_allocated() / 1e9  # convert to GB
        cached_before = torch.cuda.memory_reserved() / 1e9  # convert to GB
        print(f"Before empty_cache - Allocated: {allocated_before:.4f} GB, Cached: {cached_before:.4f} GB")

        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated() / 1e9  # convert to GB
        cached_after = torch.cuda.memory_reserved() / 1e9  # convert to GB
        print(f"After empty_cache - Allocated: {allocated_after:.4f} GB, Cached: {cached_after:.4f} GB")
