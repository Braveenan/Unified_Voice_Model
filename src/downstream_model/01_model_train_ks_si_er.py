import torch
from torch import Tensor

import os
import time
import datetime

from dataset.load_embedding import *
from dataset.custom_emb_dataloader import *
from model.downstream_model import *
from trainer.model_trainer import *
from evaluator.model_evaluator import *
from utils.constant_mapping import *

def save_to_txt(content, filename):
    with open(filename, 'a') as f:
        print(content, file=f)
        
def return_current_datetime():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return_text = f"Current date and time: {formatted_datetime}"
    return return_text

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

device = set_device(0)
transformer_layer_array = [7, 10, 20]
upstream_model_type = "wavlm_large"
upstream_model_variation = upstream_model_type.split("_")[-1]
no_of_encoders = 12 if upstream_model_variation == "base" else 24
frame_pooling_type = "mean"
layer_pooling_type = "l2"
task_type = "ks_si_er"
data_loading_percentage = 100

sub_tasks_array = task_type.split("_")
sub_frame_poolings_array = frame_pooling_type.split("_")
transformer_layer_code = "_".join(map(str, transformer_layer_array))
    
label_mapping_speechcommand = LabelMapping.LABEL2INDEX_SPEECHCOMMANDv1.value
label_mapping_voxceleb = LabelMapping.LABEL2INDEX_VOXCELEB1.value
label_mapping_iemocap = LabelMapping.LABEL2INDEX_IEMOCAP.value
    
encoder_embed_dim = 768 if upstream_model_variation == "base" else 1024
model_input_dim = encoder_embed_dim*len(sub_frame_poolings_array)
model_embedding_dim_shared = 512
    
model_embedding_dim_ks = 2000
model_embedding_dim_si = 2000
model_embedding_dim_er = 1000
    
model_output_dim_ks = len(label_mapping_speechcommand)
model_output_dim_si = len(label_mapping_voxceleb)
model_output_dim_er = len(label_mapping_iemocap)

batch_size = 2048
num_epochs = 100
learning_rate = 2.5e-3
weight_decay = 5e-8
saved_checkpoint_count = 1
patience = 1
factor = 0.5
    
dropout_prob_shared = 0.4
dropout_prob_ks = 0.55
dropout_prob_si = 0.6
dropout_prob_er = 0.5

l1_lambda = 1e-07
l2_lambda = 1e-05

root_path = "/home/braveenan/voice_dataset"
root_speechcommand = os.path.join(root_path, "SpeechCommand")
root_voxceleb = os.path.join(root_path, "VoxCeleb")
root_iemocap = os.path.join(root_path, "IEMOCAP")
    
root_emb_path = f"/home/braveenan/embedding/{upstream_model_type}/{frame_pooling_type}"
root_emb_speechcommand = os.path.join(root_emb_path, "SpeechCommand")
root_emb_voxceleb = os.path.join(root_emb_path, "VoxCeleb")
root_emb_iemocap = os.path.join(root_emb_path, "IEMOCAP")
    
current_timestamp = str(int(time.time()))
result_folder_path = f"result/{upstream_model_type}/{transformer_layer_code}/{current_timestamp}"
checkpoint_folder_path = f"checkpoint/{upstream_model_type}/{transformer_layer_code}/{current_timestamp}"

def create_file_path(upstream_model_type, task_type, folder_path, file_format):
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{upstream_model_type}_{task_type}{file_format}"
    file_path = os.path.join(folder_path, file_name)
    return file_path
    
result_text_path_all = create_file_path(upstream_model_type, task_type, result_folder_path, ".txt")
result_plot_path_all = create_file_path(upstream_model_type, task_type, result_folder_path, ".png")
result_text_path_task1 = create_file_path(upstream_model_type, sub_tasks_array[0], result_folder_path, ".txt")
result_plot_path_task1 = create_file_path(upstream_model_type, sub_tasks_array[0], result_folder_path, ".png")
result_text_path_task2 = create_file_path(upstream_model_type, sub_tasks_array[1], result_folder_path, ".txt")
result_plot_path_task2 = create_file_path(upstream_model_type, sub_tasks_array[1], result_folder_path, ".png")
result_text_path_task3 = create_file_path(upstream_model_type, sub_tasks_array[2], result_folder_path, ".txt")
result_plot_path_task3 = create_file_path(upstream_model_type, sub_tasks_array[2], result_folder_path, ".png")
model_checkpoint_path = create_file_path(upstream_model_type, task_type, checkpoint_folder_path, ".pth")
data_count_path = create_file_path(upstream_model_type, "data_count", result_folder_path, ".txt")

task_dimensions_dict = {
    "ks": (model_output_dim_ks, model_embedding_dim_ks),
    "si": (model_output_dim_si, model_embedding_dim_si),
    "er": (model_output_dim_er, model_embedding_dim_er)
}
    
dataset_dict = {
    "ks": "speechcommand",
    "si": "voxceleb",
    "er": "iemocap"
}
    
label_dict = {
    "ks": "content",
    "si": "speaker",
    "er": "emotion"
}
    
dropout_prob_dict = {
    "ks": dropout_prob_ks,
    "si": dropout_prob_si,
    "er": dropout_prob_er
}
    
dataset_root_dict = {
    "speechcommand": root_speechcommand,
    "voxceleb": root_voxceleb,
    "iemocap": root_iemocap
}
    
embedding_root_dict = {
    "speechcommand": root_emb_speechcommand,
    "voxceleb": root_emb_voxceleb,
    "iemocap": root_emb_iemocap
}
    
label_mapping_dict = {
    "speechcommand": label_mapping_speechcommand,
    "voxceleb": label_mapping_voxceleb,
    "iemocap": label_mapping_iemocap
}

model_output_dim_array = []
model_embedding_dim_array = []
dataset_array = []
label_array = []
dropout_prob_array = []
    
for task in sub_tasks_array:
    output_dim, embedding_dim = task_dimensions_dict[task]
    model_output_dim_array.append(output_dim)
    model_embedding_dim_array.append(embedding_dim)
        
    dataset_name = dataset_dict[task]
    dataset_array.append(dataset_name)
        
    label_name = label_dict[task]
    label_array.append(label_name)
        
    dropout_prob_name = dropout_prob_dict[task]
    dropout_prob_array.append(dropout_prob_name)
    
current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, result_text_path_all)
save_to_txt(current_datetime, result_text_path_task1)
save_to_txt(current_datetime, result_text_path_task2)
save_to_txt(current_datetime, result_text_path_task3)

task1_name = KeywordMapping.get_task_name(sub_tasks_array[0])
task1_text = f"Task1 name: {task1_name}"
save_to_txt(task1_text, result_text_path_task1)
task2_name = KeywordMapping.get_task_name(sub_tasks_array[1])
task2_text = f"Task2 name: {task2_name}"
save_to_txt(task2_text, result_text_path_task2)
task3_name = KeywordMapping.get_task_name(sub_tasks_array[2])
task3_text = f"Task3 name: {task3_name}"
save_to_txt(task3_text, result_text_path_task3)
tasks_name = f"{task1_name}, {task2_name} and {task3_name}"
tasks_text = f"Tasks name: {tasks_name}"
print(tasks_text)
save_to_txt(tasks_text, result_text_path_all)

upstreammodel_text = f"Upstream model type: {upstream_model_type}"
print(upstreammodel_text)
save_to_txt(upstreammodel_text, result_text_path_all)
save_to_txt(upstreammodel_text, result_text_path_task1)
save_to_txt(upstreammodel_text, result_text_path_task2)
save_to_txt(upstreammodel_text, result_text_path_task3)
    
transformer_text = f"Transformer layers: {','.join(map(str, transformer_layer_array))}"
print(transformer_text)
save_to_txt(transformer_text, result_text_path_all)
save_to_txt(transformer_text, result_text_path_task1)
save_to_txt(transformer_text, result_text_path_task2)
save_to_txt(transformer_text, result_text_path_task3)
    
layer_pooling_text = f"Layer pooling type: {layer_pooling_type}"
print(layer_pooling_text)
save_to_txt(layer_pooling_text, result_text_path_all)
save_to_txt(layer_pooling_text, result_text_path_task1)
save_to_txt(layer_pooling_text, result_text_path_task2)
save_to_txt(layer_pooling_text, result_text_path_task3)

loader = LoadEmbedding(
    dataset_root_dict=dataset_root_dict,
    embedding_root_dict=embedding_root_dict,
    label_mapping_dict=label_mapping_dict,
    frame_pooling_type = frame_pooling_type,
    upstream_model_type=upstream_model_type,
    transformer_layer_array=transformer_layer_array,
    device=device
)
    
dataset_text = f"Dataset name: {dataset_array[0]}, {dataset_array[1]} and {dataset_array[2]}"
print(dataset_text)
save_to_txt(dataset_text, result_text_path_all)
save_to_txt(dataset_text, result_text_path_task1)
save_to_txt(dataset_text, result_text_path_task2)
save_to_txt(dataset_text, result_text_path_task3)
    
dataset_loading_code = "_".join(dataset_array)
training_data, validation_data, testing_data = loader.load_embedding(dataset_loading_code, data_loading_percentage)
dataset_length_text = f"No of training data samples: {len(training_data)} \nNo of validation data samples: {len(validation_data)}"
print(dataset_length_text)
save_to_txt(dataset_length_text, result_text_path_all)
save_to_txt(dataset_length_text, result_text_path_task1)
save_to_txt(dataset_length_text, result_text_path_task2)
save_to_txt(dataset_length_text, result_text_path_task3)

train_dataloader = CustomEmbDataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
val_dataloader = CustomEmbDataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

model = DownstreamTripleTaskModel(model_input_dim, model_output_dim_array, model_embedding_dim_shared, model_embedding_dim_array, layer_pooling_type, dropout_prob_shared, dropout_prob_array)
model.to(device)
print(model)

optimizer_parameters = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay
}
scheduler_parameters = {
    "patience": patience, 
    "factor": factor
}

trainer = TripleTasksModelTrainer(model, optimizer_parameters, scheduler_parameters, device, label_array, num_epochs, saved_checkpoint_count, l1_lambda, l2_lambda)
trainer.train_dataloader = train_dataloader
trainer.test_dataloader = val_dataloader
trainer.data_count_path = data_count_path
trainer.result_text_path_all = result_text_path_all
trainer.result_plot_path_all = result_plot_path_all
trainer.result_text_path_task1 = result_text_path_task1
trainer.result_plot_path_task1 = result_plot_path_task1
trainer.result_text_path_task2 = result_text_path_task2
trainer.result_plot_path_task2 = result_plot_path_task2
trainer.result_text_path_task3 = result_text_path_task3
trainer.result_plot_path_task3 = result_plot_path_task3
trainer.model_checkpoint_path = model_checkpoint_path
trainer.plot_title_all = tasks_name
trainer.plot_title_task1 = task1_name
trainer.plot_title_task2 = task2_name
trainer.plot_title_task3 = task3_name
    
# Train the model
trainer.train()
    
# Plot metrics separately when needed
trainer.plot_metrics()

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, result_text_path_all)
save_to_txt(current_datetime, result_text_path_task1)
save_to_txt(current_datetime, result_text_path_task2)
save_to_txt(current_datetime, result_text_path_task3)

best_checkpoint_path = model_checkpoint_path.replace(".pth", "_best.pth")
print(best_checkpoint_path)

opt_checkpoint_path = model_checkpoint_path.replace(".pth", "_opt.pth")
print(opt_checkpoint_path)

best_file_name = best_checkpoint_path.replace(".pth", "")
best_file_name = best_file_name.replace("checkpoint/", "result/")
best_text_path = f"{best_file_name}_eval.txt"

opt_file_name = opt_checkpoint_path.replace(".pth", "")
opt_file_name = opt_file_name.replace("checkpoint/", "result/")
opt_text_path = f"{opt_file_name}_eval.txt"

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, best_text_path)
save_to_txt(current_datetime, opt_text_path)

save_to_txt(tasks_text, best_text_path)
save_to_txt(upstreammodel_text, best_text_path)
save_to_txt(transformer_text, best_text_path)
save_to_txt(layer_pooling_text, best_text_path)

save_to_txt(tasks_text, opt_text_path)
save_to_txt(upstreammodel_text, opt_text_path)
save_to_txt(transformer_text, opt_text_path)
save_to_txt(layer_pooling_text, opt_text_path)

print(dataset_text)
save_to_txt(dataset_text, best_text_path)
save_to_txt(dataset_text, opt_text_path)

dataset_length_text = f"No of testing data samples: {len(testing_data)}"
print(dataset_length_text)
save_to_txt(dataset_length_text, best_text_path)
save_to_txt(dataset_length_text, opt_text_path)

evaluator = TripleTaskModelEvaluator(model, best_checkpoint_path, testing_data, device, label_array)
evaluator.get_loss_and_accuracy(best_text_path)
evaluator.get_labels_and_predictions(best_file_name)

evaluator = TripleTaskModelEvaluator(model, opt_checkpoint_path, testing_data, device, label_array)
evaluator.get_loss_and_accuracy(opt_text_path)
evaluator.get_labels_and_predictions(opt_file_name)

current_datetime = return_current_datetime()
print(current_datetime)
save_to_txt(current_datetime, best_text_path)
save_to_txt(current_datetime, opt_text_path)