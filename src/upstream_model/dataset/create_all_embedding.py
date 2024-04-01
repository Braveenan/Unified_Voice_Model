import torch
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset, Subset

from torchaudio.datasets import SPEECHCOMMANDS, VoxCeleb1Identification, IEMOCAP
from torchaudio.datasets.utils import _load_waveform

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Any
from scipy import stats

import os

# Speech Commands dataset pre-processing
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.01"
class SPEECHCOMMANDSEmbedding(SPEECHCOMMANDS):
    def __init__(
        self,
        root: Union[str, Path],
        frame_pooling_type: str = None,
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        model: torch.nn.Module = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download, subset)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.device = device
        
    def _get_vector_after_pooling(self, data, frame_pooling_type):
        def mean_pooling(x):
            return x.mean(dim=1)
        def std_pooling(x):
            return x.std(dim=1)
        def first_pooling(x):
            return x[:, 0, :]
        def last_pooling(x):
            return x[:, -1, :]
        def max_pooling(x):
            return x.max(dim=1).values
        def min_pooling(x):
            return x.min(dim=1).values
        def skew_pooling(x):
            return torch.tensor(stats.skew(x.cpu().numpy(), axis=1), dtype=x.dtype, device=x.device)

        pooling_operations = {
            "mean": mean_pooling,
            "std": std_pooling,
            "first": first_pooling,
            "last": last_pooling,
            "max": max_pooling,
            "min": min_pooling,
            "skew": skew_pooling,
        }

        sub_pooling_array = frame_pooling_type.split("_")
        result = []
        for component in sub_pooling_array:
            if component in pooling_operations:
                result.append(pooling_operations[component](data))
            else:
                raise ValueError("Invalid frame_pooling_type: " + frame_pooling_type)
        return torch.cat(result, dim=1)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        
        wavpath = metadata[0] 
        label = str(metadata[2])
        
        if self.device is None:
            self.device = torch.device("cpu")
        
        waveform = waveform.to(self.device)
        self.model = self.model.to(self.device)
        
        wavlength = waveform.size(1)
        embedding = None
        emblength = None
            
        if self.model is not None:
            with torch.no_grad():
                features, _ = self.model.extract_features(waveform)
                embedding = torch.stack(features).squeeze(dim=1)
                emblength = embedding.size(1)
                embedding = self._get_vector_after_pooling(embedding, self.frame_pooling_type)

        return embedding, emblength, wavlength, label, wavpath

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        return self.get_metadata(n)
    
# Voxceleb dataset pre-processing
_IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
class VoxCeleb1Embedding(VoxCeleb1Identification):
    def __init__(
        self, 
        root: Union[str, Path],
        frame_pooling_type: str = None,
        subset: str = "train", 
        model: torch.nn.Module = None,
        download: bool = False,
        device: torch.device = None,
    ) -> None:
        super().__init__(root, subset, _IDEN_SPLIT_URL, download)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.device = device
        
    def _get_vector_after_pooling(self, data, frame_pooling_type):
        def mean_pooling(x):
            return x.mean(dim=1)
        def std_pooling(x):
            return x.std(dim=1)
        def first_pooling(x):
            return x[:, 0, :]
        def last_pooling(x):
            return x[:, -1, :]
        def max_pooling(x):
            return x.max(dim=1).values
        def min_pooling(x):
            return x.min(dim=1).values
        def skew_pooling(x):
            return torch.tensor(stats.skew(x.cpu().numpy(), axis=1), dtype=x.dtype, device=x.device)

        pooling_operations = {
            "mean": mean_pooling,
            "std": std_pooling,
            "first": first_pooling,
            "last": last_pooling,
            "max": max_pooling,
            "min": min_pooling,
            "skew": skew_pooling,
        }

        sub_pooling_array = frame_pooling_type.split("_")
        result = []
        for component in sub_pooling_array:
            if component in pooling_operations:
                result.append(pooling_operations[component](data))
            else:
                raise ValueError("Invalid frame_pooling_type: " + frame_pooling_type)
        return torch.cat(result, dim=1)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        
        wavpath = metadata[0] 
        label = str(metadata[2])
        
        if self.device is None:
            self.device = torch.device("cpu")
        
        waveform = waveform.to(self.device)
        self.model = self.model.to(self.device)
        
        wavlength = waveform.size(1)
        embedding = None
        emblength = None
            
        if self.model is not None:
            with torch.no_grad():
                features, _ = self.model.extract_features(waveform)
                embedding = torch.stack(features).squeeze(dim=1)
                emblength = embedding.size(1)
                embedding = self._get_vector_after_pooling(embedding, self.frame_pooling_type)

        return embedding, emblength, wavlength, label, wavpath 

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        return self.get_metadata(n)

# IEMOCAP dataset pre-processing
class IEMOCAPEmbedding(IEMOCAP):
    def __init__(
        self,
        root: Union[str, Path],
        frame_pooling_type: str = None,
        sessions: Tuple[str] = (1, 2, 3, 4, 5),
        utterance_type: Optional[str] = None,
        model: torch.nn.Module = None,
        device: torch.device = None,
    ) -> None:
        super().__init__(root, sessions, utterance_type)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.device = device
        
    def _get_vector_after_pooling(self, data, frame_pooling_type):
        def mean_pooling(x):
            return x.mean(dim=1)
        def std_pooling(x):
            return x.std(dim=1)
        def first_pooling(x):
            return x[:, 0, :]
        def last_pooling(x):
            return x[:, -1, :]
        def max_pooling(x):
            return x.max(dim=1).values
        def min_pooling(x):
            return x.min(dim=1).values
        def skew_pooling(x):
            return torch.tensor(stats.skew(x.cpu().numpy(), axis=1), dtype=x.dtype, device=x.device)

        pooling_operations = {
            "mean": mean_pooling,
            "std": std_pooling,
            "first": first_pooling,
            "last": last_pooling,
            "max": max_pooling,
            "min": min_pooling,
            "skew": skew_pooling,
        }

        sub_pooling_array = frame_pooling_type.split("_")
        result = []
        for component in sub_pooling_array:
            if component in pooling_operations:
                result.append(pooling_operations[component](data))
            else:
                raise ValueError("Invalid frame_pooling_type: " + frame_pooling_type)
        return torch.cat(result, dim=1)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)

        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        
        wavpath = metadata[0] 
        label = str(metadata[3])
        
        if self.device is None:
            self.device = torch.device("cpu")
                
        waveform = waveform.to(self.device)
        self.model = self.model.to(self.device)
        
        wavlength = waveform.size(1)
        embedding = None
        emblength = None
            
        if self.model is not None:
            with torch.no_grad():
                features, _ = self.model.extract_features(waveform)
                embedding = torch.stack(features).squeeze(dim=1)
                emblength = embedding.size(1)
                embedding = self._get_vector_after_pooling(embedding, self.frame_pooling_type)

        return embedding, emblength, wavlength, label, wavpath

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str, str]:  
        return self.get_metadata(n)


