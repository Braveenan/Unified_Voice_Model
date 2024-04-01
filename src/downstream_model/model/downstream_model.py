import torch
import torch.nn as nn
        
class TripleClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


class DownstreamTripleTaskModel(nn.Module):
    def __init__(self, input_dim, output_dim_array, embedding_dim_shared, embedding_dim_array, layer_pooling_type, dropout_prob_shared, dropout_prob_array):
        super(DownstreamTripleTaskModel, self).__init__()
        
        output_dim_task1 = output_dim_array[0]
        embedding_dim_task1 = embedding_dim_array[0]
        output_dim_task2 = output_dim_array[1]
        embedding_dim_task2 = embedding_dim_array[1]
        output_dim_task3 = output_dim_array[2]
        embedding_dim_task3 = embedding_dim_array[2]
        
        sub_layer_pooling_array = layer_pooling_type.split("_")
        no_of_layer_pooling = len(sub_layer_pooling_array)
        
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared)
        self.dropout_shared = nn.Dropout(p=dropout_prob_shared)
        self.hidden_layer1 = nn.Linear(embedding_dim_shared*no_of_layer_pooling, embedding_dim_task1)
        self.dropout_task1 = nn.Dropout(p=dropout_prob_array[0])
        self.hidden_layer2 = nn.Linear(embedding_dim_shared*no_of_layer_pooling, embedding_dim_task2)
        self.dropout_task2 = nn.Dropout(p=dropout_prob_array[1])
        self.hidden_layer3 = nn.Linear(embedding_dim_shared*no_of_layer_pooling, embedding_dim_task3)
        self.dropout_task3 = nn.Dropout(p=dropout_prob_array[2])
        self.classifier_task1 = nn.Linear(embedding_dim_task1, output_dim_task1)
        self.classifier_task2 = nn.Linear(embedding_dim_task2, output_dim_task2)
        self.classifier_task3 = nn.Linear(embedding_dim_task3, output_dim_task3)
        
        self.layer_pooling_type = layer_pooling_type
    
    def _layer_pooling(self, data, layer_pooling_type):
        def mean_pooling(x):
            return x.mean(dim=1)

        def std_pooling(x):
            return x.std(dim=1)

        def max_pooling(x):
            return x.max(dim=1).values

        def min_pooling(x):
            return x.min(dim=1).values

        pooling_operations = {
            "mean": mean_pooling,
            "std": std_pooling,
            "max": max_pooling,
            "min": min_pooling,
        }

        sub_layer_pooling_array = layer_pooling_type.split("_")
        result = []
        for component in sub_layer_pooling_array:
            if component in pooling_operations:
                result.append(pooling_operations[component](data))
            else:
                raise ValueError("Invalid layer pooling type: " + layer_pooling_type)
        return torch.cat(result, dim=1)

    def forward(self, input_seq):
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self._layer_pooling(embedding_shared, self.layer_pooling_type)
        embedding_shared = self.dropout_shared(embedding_shared)
        
        embedding_task1 = self.hidden_layer1(embedding_shared)
        embedding_task1 = self.dropout_task1(embedding_task1)
        embedding_task2 = self.hidden_layer2(embedding_shared)
        embedding_task2 = self.dropout_task2(embedding_task2)
        embedding_task3 = self.hidden_layer3(embedding_shared)
        embedding_task3 = self.dropout_task3(embedding_task3)
        
        logits_task1 = self.classifier_task1(embedding_task1)
        logits_task2 = self.classifier_task2(embedding_task2)
        logits_task3 = self.classifier_task3(embedding_task3)

        _, prediction_task1 = torch.max(logits_task1, 1)
        _, prediction_task2 = torch.max(logits_task2, 1)
        _, prediction_task3 = torch.max(logits_task3, 1)

        logits = (logits_task1, logits_task2, logits_task3)
        prediction = (prediction_task1, prediction_task2, prediction_task3)
        
        return TripleClassifierOutput(
            logits = logits,
            prediction = prediction
        )
    
    def get_all_embeddings(self, input_seq):
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self._layer_pooling(embedding_shared, self.layer_pooling_type)
        embedding_task1 = self.hidden_layer1(embedding_shared)
        embedding_task2 = self.hidden_layer2(embedding_shared)
        embedding_task3 = self.hidden_layer3(embedding_shared)
        return embedding_shared, embedding_task1, embedding_task2, embedding_task3

    