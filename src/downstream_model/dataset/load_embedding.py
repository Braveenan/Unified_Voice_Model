from dataset.preprocess_embedding import *

class LoadEmbedding:
    def __init__(
        self,
        dataset_root_dict=None,
        embedding_root_dict=None,
        label_mapping_dict=None,
        upstream_model_type=None,
        frame_pooling_type=None,
        transformer_layer_array=None,
        device=None
    ):
        self.root_speechcommand = dataset_root_dict["speechcommand"]
        self.root_voxceleb = dataset_root_dict["voxceleb"]
        self.root_iemocap = dataset_root_dict["iemocap"]
        self.root_emb_speechcommand = embedding_root_dict["speechcommand"]
        self.root_emb_voxceleb = embedding_root_dict["voxceleb"]
        self.root_emb_iemocap = embedding_root_dict["iemocap"]
        self.label_mapping_speechcommand = label_mapping_dict["speechcommand"]
        self.label_mapping_voxceleb = label_mapping_dict["voxceleb"]
        self.label_mapping_iemocap = label_mapping_dict["iemocap"]
        self.upstream_model_type = upstream_model_type
        self.frame_pooling_type = frame_pooling_type
        self.transformer_layer_array = transformer_layer_array
        self.device = device
    
    def load_embedding(self, dataset_id, subset_percentage=None):
        load_dataset_dict = {
            "speechcommand": self._load_speechcommand,
            "voxceleb": self._load_voxceleb,
            "iemocap": self._load_iemocap,
        }
        
        sub_dataset_id_array = dataset_id.split("_")
        training_data_array = []
        validation_data_array = []
        testing_data_array = []
        for dataset_id in sub_dataset_id_array:
            if dataset_id in load_dataset_dict:
                training_data, validation_data, testing_data = load_dataset_dict[dataset_id]()
                training_data_array.append(training_data)
                validation_data_array.append(validation_data)
                testing_data_array.append(testing_data)
            else:
                raise ValueError("Invalid dataset id: " + dataset_id)
                
        training_data = CombinedDataset(training_data_array)
        validation_data = CombinedDataset(validation_data_array)
        testing_data = CombinedDataset(testing_data_array)
        
        if subset_percentage is not None:
            training_data = PercentageSubset(training_data, subset_percentage)
            validation_data = PercentageSubset(validation_data, subset_percentage)
            testing_data = PercentageSubset(testing_data, subset_percentage)

        return training_data, validation_data, testing_data

    def _load_speechcommand(self):
        root_speechcommand = self.root_speechcommand
        root_emb_speechcommand = self.root_emb_speechcommand
        label_mapping_speechcommand = self.label_mapping_speechcommand
        frame_pooling_type = self.frame_pooling_type
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device
        
        if root_speechcommand is None:
            raise ValueError("Speechcommand dataset root not provided")
            
        if root_emb_speechcommand is None:
            raise ValueError("Speechcommand dataset embedding root not provided")
            
        if label_mapping_speechcommand is None:
            raise ValueError("Speechcommand dataset label mapping dictionary not provided")
        
        speechcommand_train_data = SPEECHCOMMANDSEmbedding (
            root = root_speechcommand,
            root_embedding = root_emb_speechcommand,
            frame_pooling_type=frame_pooling_type,
            url = "speech_commands_v0.01",
            subset = "training",
            download=False,
            label_mapping=label_mapping_speechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )
        
        speechcommand_val_data = SPEECHCOMMANDSEmbedding (
            root = root_speechcommand,
            root_embedding = root_emb_speechcommand,
            frame_pooling_type=frame_pooling_type,
            url = "speech_commands_v0.01",
            subset = "validation",
            download=False,
            label_mapping=label_mapping_speechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )
        
        speechcommand_test_data = SPEECHCOMMANDSEmbedding (
            root = root_speechcommand,
            root_embedding = root_emb_speechcommand,
            frame_pooling_type=frame_pooling_type,
            url = "speech_commands_v0.01",
            subset = "testing",
            download=False,
            label_mapping=label_mapping_speechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )
        
        return speechcommand_train_data, speechcommand_val_data, speechcommand_test_data

    def _load_voxceleb(self):
        root_voxceleb = self.root_voxceleb
        root_emb_voxceleb = self.root_emb_voxceleb
        label_mapping_voxceleb = self.label_mapping_voxceleb
        frame_pooling_type = self.frame_pooling_type
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device
        
        if root_voxceleb is None:
            raise ValueError("Voxceleb dataset root not provided")
            
        if root_emb_voxceleb is None:
            raise ValueError("Voxceleb dataset embeddiroot not provided")
            
        if label_mapping_voxceleb is None:
            raise ValueError("Voxceleb dataset label mapping dictionary not provided")
        
        voxceleb_train_data = VoxCeleb1Embedding (
            root=root_voxceleb,
            root_embedding = root_emb_voxceleb,
            frame_pooling_type=frame_pooling_type,
            subset = 'train',
            download=False,
            label_mapping=label_mapping_voxceleb,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )
        
        voxceleb_val_data = VoxCeleb1Embedding (
            root=root_voxceleb,
            root_embedding = root_emb_voxceleb,
            frame_pooling_type=frame_pooling_type,
            subset = 'dev',
            download=False,
            label_mapping=label_mapping_voxceleb,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )
        
        voxceleb_test_data = VoxCeleb1Embedding (
            root=root_voxceleb,
            root_embedding = root_emb_voxceleb,
            frame_pooling_type=frame_pooling_type,
            subset = 'test',
            download=False,
            label_mapping=label_mapping_voxceleb,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )
        
        return voxceleb_train_data, voxceleb_val_data, voxceleb_test_data

    def _load_iemocap(self):
        root_iemocap = self.root_iemocap
        root_emb_iemocap = self.root_emb_iemocap
        label_mapping_iemocap = self.label_mapping_iemocap
        frame_pooling_type = self.frame_pooling_type
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device
        
        if root_iemocap is None:
            raise ValueError("IEMOCAP dataset root not provided")
            
        if root_emb_iemocap is None:
            raise ValueError("IEMOCAP dataset embedding root not provided")
            
        if label_mapping_iemocap is None:
            raise ValueError("IEMOCAP dataset label mapping dictionary not provided")
        
        iemocap_total_data = IEMOCAPEmbedding (
            root = root_iemocap,
            root_embedding = root_emb_iemocap,
            frame_pooling_type=frame_pooling_type,
            sessions = ("1", "2", "3", "4", "5"),
            label_mapping = label_mapping_iemocap,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
        )

        iemocap_total_samples = len(iemocap_total_data)

        # Create indices for the split
        indices = list(range(iemocap_total_samples))
        val_indices = indices[4::10]  
        test_indices = indices[9::10]  
        train_indices = [idx for idx in indices if idx not in val_indices and idx not in test_indices]

        iemocap_train_data = torch.utils.data.Subset(iemocap_total_data, train_indices)
        iemocap_val_data = torch.utils.data.Subset(iemocap_total_data, val_indices)
        iemocap_test_data = torch.utils.data.Subset(iemocap_total_data, test_indices)
        
        return iemocap_train_data, iemocap_val_data, iemocap_test_data
