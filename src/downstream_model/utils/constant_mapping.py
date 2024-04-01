from enum import Enum
import torchaudio

class LabelMapping(Enum):
    LABEL2INDEX_SPEECHCOMMANDv1 = {
        '_silence_': 11,
        '_unknown_': 10,
        'down': 3,
        'go': 9,
        'left': 4,
        'no': 1,
        'off': 7,
        'on': 6,
        'right': 5,
        'stop': 8,
        'up': 2,
        'yes': 0
    }
    INDEX2LABEL_SPEECHCOMMANDv1 = {v: k for k, v in LABEL2INDEX_SPEECHCOMMANDv1.items()}
    
    LABEL2INDEX_VOXCELEB1 = {str(i): i-1 for i in range(1, 1252)}
    INDEX2LABEL_VOXCELEB1 = {v: k for k, v in LABEL2INDEX_VOXCELEB1.items()}
    
    LABEL2INDEX_IEMOCAP = {'ang': 2, 'hap': 1, 'neu': 0, 'sad': 3}
    INDEX2LABEL_IEMOCAP = {v: k for k, v in LABEL2INDEX_IEMOCAP.items()}
    
class ModelMapping:
    wav2vec2_base = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec2_large = torchaudio.pipelines.WAV2VEC2_LARGE
    hubert_base = torchaudio.pipelines.HUBERT_BASE
    hubert_large = torchaudio.pipelines.HUBERT_LARGE
    # wavlm_base = torchaudio.pipelines.WAVLM_BASE
    # wavlm_large = torchaudio.pipelines.WAVLM_LARGE
    
    @classmethod
    def get_model_bundle(cls, key):
        return getattr(cls, key, None)
    
class KeywordMapping:
    ks = "Keyword Spotting"
    si = "Speaker Identification"
    er = "Emotion Recognition"
    
    @classmethod
    def get_task_name(cls, key):
        return getattr(cls, key, None)
    