import torch
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset, Subset

from torchaudio.datasets import SPEECHCOMMANDS, VoxCeleb1Identification, IEMOCAP
from torchaudio.datasets.utils import _load_waveform

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Any

import os
import re

# Speech Commands dataset pre-processing
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.01"
class SPEECHCOMMANDSEmbedding(SPEECHCOMMANDS):
    def __init__(
        self,
        root: Union[str, Path],
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        upstream_model_type: str,
        label_mapping: Dict[str, int],
        transformer_layer_array: List[int] = None, 
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download, subset)
        self.root_embedding = root_embedding
        self.frame_pooling_type = frame_pooling_type
        self.label_mapping = label_mapping 
        self.upstream_model_type = upstream_model_type
        self.transformer_layer_array = transformer_layer_array
        self.device = device

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, int]:
        metadata = super().get_metadata(n)
        sub_pooling_array = self.frame_pooling_type.split("_")
        wav2vec_array = []
        for pooling_component in sub_pooling_array:
            embpath = metadata[0].replace(".wav", f"_{self.upstream_model_type}_{pooling_component}.pt")
            root_embpath = self.root_embedding.replace(f"/{self.frame_pooling_type}/", f"/{pooling_component}/")
            root_embpath = os.path.join(root_embpath, embpath)
            # wav2vec_component = torch.load(root_embpath, map_location=torch.device("cpu"))
            try:
                wav2vec_component = torch.load(root_embpath, map_location=torch.device("cpu"))
            except EOFError as e:
                print(f"Error loading data from {root_embpath}: {e}")
            wav2vec_array.append(wav2vec_component)
        wav2vec = torch.cat(wav2vec_array, dim=1)
        
        if self.transformer_layer_array is not None:
            layer_indices = [idx - 1 for idx in self.transformer_layer_array]
            wav2vec = wav2vec[layer_indices, :]
        
        label = metadata[2]
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
        else:
            label_index = 10
            
        content_index = label_index
        speaker_index = -1
        emotion_index = -1

        return wav2vec, content_index, speaker_index, emotion_index

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, int]:
        return self.get_metadata(n)
    
# Voxceleb dataset pre-processing
_IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
class VoxCeleb1Embedding(VoxCeleb1Identification):
    def __init__(
        self, 
        root: Union[str, Path], 
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        upstream_model_type: str,
        label_mapping: Dict[str, int],
        subset: str = "train", 
        transformer_layer_array: List[int] = None,
        download: bool = False,
        device: torch.device = None,
    ) -> None:
        super().__init__(root, subset, _IDEN_SPLIT_URL, download)
        self.root_embedding = root_embedding
        self.frame_pooling_type = frame_pooling_type
        self.label_mapping = label_mapping
        self.upstream_model_type = upstream_model_type
        self.transformer_layer_array = transformer_layer_array
        self.device = device

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, int]:
        metadata = super().get_metadata(n)
        sub_pooling_array = self.frame_pooling_type.split("_")
        wav2vec_array = []
        for pooling_component in sub_pooling_array:
            embpath = metadata[0].replace(".wav", f"_{self.upstream_model_type}_{pooling_component}.pt")
            root_embpath = self.root_embedding.replace(f"/{self.frame_pooling_type}/", f"/{pooling_component}/")
            root_embpath = os.path.join(root_embpath, embpath)
            # wav2vec_component = torch.load(root_embpath, map_location=torch.device("cpu"))
            try:
                wav2vec_component = torch.load(root_embpath, map_location=torch.device("cpu"))
            except EOFError as e:
                print(f"Error loading data from {root_embpath}: {e}")
            wav2vec_array.append(wav2vec_component)
        wav2vec = torch.cat(wav2vec_array, dim=1)
        
        if self.transformer_layer_array is not None:
            layer_indices = [idx - 1 for idx in self.transformer_layer_array]
            wav2vec = wav2vec[layer_indices, :]
            
        label = str(metadata[2]) 
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
            
        content_index = -1
        speaker_index = label_index
        emotion_index = -1

        return wav2vec, content_index, speaker_index, emotion_index 

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, int]:
        return self.get_metadata(n)

# IEMOCAP dataset pre-processing
class IEMOCAPEmbedding(IEMOCAP):
    def __init__(
        self,
        root: Union[str, Path],
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        upstream_model_type: str,
        sessions: List[str],
        transformer_layer_array: List[int] = None,
        utterance_type: Optional[str] = None,
        label_mapping: Dict[str, int] = None,
        device: torch.device = None,
    ) -> None:

        sessions, speakers = self.process_sessions(sessions)
    
        super().__init__(root, sessions, utterance_type)
        self.root_embedding = root_embedding
        self.frame_pooling_type = frame_pooling_type
        self.label_mapping = label_mapping
        self.upstream_model_type = upstream_model_type
        self.transformer_layer_array = transformer_layer_array
        self.data = [wav_stem for wav_stem in self.data if self.mapping[wav_stem]["label"] != "fru"]
        self.data = [wav_stem for wav_stem in self.data if self.mapping[wav_stem]["path"].split("/")[3].split("_")[0] in speakers]
        self.device = device
            
    def process_sessions(self, input_array):
        session_id_array = []
        speaker_id_array = []  
        
        for element in input_array:
            match = re.search(r'\d+', element)
            numeric_part = match.group() if match else ''
            session_id_array.append(numeric_part)
            processed_output = 'Ses0' + element
                
            if numeric_part != element:
                speaker_id_array.append(processed_output)
            else:
                speaker_id_array.extend([processed_output + 'F', processed_output + 'M'])
        
        return tuple(session_id_array), tuple(speaker_id_array)
    
    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, int]:
        metadata = super().get_metadata(n)
        sub_pooling_array = self.frame_pooling_type.split("_")
        wav2vec_array = []
        for pooling_component in sub_pooling_array:
            embpath = metadata[0].replace(".wav", f"_{self.upstream_model_type}_{pooling_component}.pt")
            root_embpath = self.root_embedding.replace(f"/{self.frame_pooling_type}/", f"/{pooling_component}/")
            root_embpath = os.path.join(root_embpath, embpath)
            # wav2vec_component = torch.load(root_embpath, map_location=torch.device("cpu"))
            try:
                wav2vec_component = torch.load(root_embpath, map_location=torch.device("cpu"))
            except EOFError as e:
                print(f"Error loading data from {root_embpath}: {e}")
            wav2vec_array.append(wav2vec_component)
        wav2vec = torch.cat(wav2vec_array, dim=1)
        
        if self.transformer_layer_array is not None:
            layer_indices = [idx - 1 for idx in self.transformer_layer_array]
            wav2vec = wav2vec[layer_indices, :]
        
        label = metadata[3]
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
        elif label == "exc":
            label_index = self.label_mapping['hap']

        content_index = -1
        speaker_index = -1 
        emotion_index = label_index

        return wav2vec, content_index, speaker_index, emotion_index

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, int]:  
        return self.get_metadata(n)

# Combining multiple datasets
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.total_length = sum(len(dataset) for dataset in datasets)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)

# Taking subset of data        
class PercentageSubset(Subset):
    def __init__(self, dataset: Dataset, percentage: int) -> None:
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100")

        total_samples = len(dataset)
        num_samples = total_samples * percentage // 100

        indices = list(range(0, total_samples, max(1, total_samples // num_samples)))

        super().__init__(dataset, indices)
