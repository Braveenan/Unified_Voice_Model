import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.custom_emb_dataloader import *
import pandas as pd
import csv
       
class TripleTaskEvaluatorOutput:
    def __init__(self, avg_loss_all=None, accuracy_all=None, avg_loss_task1=None, accuracy_task1=None, avg_loss_task2=None, accuracy_task2=None, avg_loss_task3=None, accuracy_task3=None):
        self.avg_loss_all = avg_loss_all
        self.accuracy_all = accuracy_all
        self.avg_loss_task1 = avg_loss_task1
        self.accuracy_task1 = accuracy_task1
        self.avg_loss_task2 = avg_loss_task2
        self.accuracy_task2 = accuracy_task2
        self.avg_loss_task3 = avg_loss_task3
        self.accuracy_task3 = accuracy_task3

        
class TripleTaskBatchProcessOutput:
    def __init__(self, loss_all=None, batch_all=None, correct_count_all=None, samples_count_all=None, loss_task1=None, batch_task1=None, correct_count_task1=None, samples_count_task1=None, loss_task2=None, batch_task2=None, correct_count_task2=None, samples_count_task2=None, loss_task3=None, batch_task3=None, correct_count_task3=None, samples_count_task3=None, labels_task1=None, predictions_task1=None, labels_task2=None, predictions_task2=None, labels_task3=None, predictions_task3=None):
        self.loss_all = loss_all
        self.batch_all = batch_all
        self.correct_count_all = correct_count_all
        self.samples_count_all = samples_count_all
        self.loss_task1 = loss_task1
        self.batch_task1 = batch_task1
        self.correct_count_task1 = correct_count_task1
        self.samples_count_task1 = samples_count_task1
        self.loss_task2 = loss_task2
        self.batch_task2 = batch_task2
        self.correct_count_task2 = correct_count_task2
        self.samples_count_task2 = samples_count_task2
        self.loss_task3 = loss_task3
        self.batch_task3 = batch_task3
        self.correct_count_task3 = correct_count_task3
        self.samples_count_task3 = samples_count_task3
        self.labels_task1 = labels_task1
        self.predictions_task1 = predictions_task1
        self.labels_task2 = labels_task2
        self.predictions_task2 = predictions_task2
        self.labels_task3 = labels_task3
        self.predictions_task3 = predictions_task3
        
           
class TripleTaskModelEvaluator:
    def __init__(self, model, model_checkpoint_path, dataset, device, label_key_array):
        self.device = device
        self.label_key_array = label_key_array
        
        self.dataloader = CustomEmbDataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        if model_checkpoint_path is not None:
            checkpoint = torch.load(model_checkpoint_path, map_location=torch.device(device))
            model.load_state_dict(checkpoint)
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()
        
    def _get_predicted_and_count(self, prediction, labels):
        mask = labels != -1
        labels_masked = labels[mask]
        prediction_masked = prediction[mask]
        correct_count = (prediction_masked == labels_masked).sum().item()
        samples_count = labels_masked.size(0)
        return labels_masked, prediction_masked, correct_count, samples_count

    def _calculate_total_loss(self, loss_task, loss_all, loss_weight):
        batch_task = int(loss_task is not None)
        if loss_task is None:
            loss_task = torch.tensor(0)
            loss_task.to(self.device)
        else:
            loss_all += loss_task*loss_weight
        return loss_task, loss_all, batch_task

    def _calculate_batch_loss(self, logits, labels, loss_fn):
        loss = None
        if labels is not None:
            mask = labels != -1
            labels_masked = labels[mask]
            if labels_masked.size(0) != 0:
                logits_masked = logits[mask, :]
                loss = loss_fn(logits_masked, labels_masked)
        return loss

    def _process_batch(self, batch):
        input_seq = batch[0].to(self.device)
        labels_task1 = batch[1].to(self.device)
        labels_task2 = batch[2].to(self.device)
        labels_task3 = batch[3].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_seq=input_seq)
            logits_task1 = outputs.logits[0]
            logits_task2 = outputs.logits[1]
            logits_task3 = outputs.logits[2]
            prediction_task1 = outputs.prediction[0]
            prediction_task2 = outputs.prediction[1]
            prediction_task3 = outputs.prediction[2]
            
            loss_all = 0
            batch_all = 1
            
            num_tasks = 3
            loss_weight1 = 1.0 / num_tasks
            loss_weight2 = 1.0 / num_tasks
            loss_weight3 = 1.0 / num_tasks
                
            loss_task1 = self._calculate_batch_loss(logits_task1, labels_task1, self.loss_fn)
            loss_task2 = self._calculate_batch_loss(logits_task2, labels_task2, self.loss_fn)
            loss_task3 = self._calculate_batch_loss(logits_task3, labels_task3, self.loss_fn)
        
            loss_task1, loss_all, batch_task1 = self._calculate_total_loss(loss_task1, loss_all, loss_weight1)
            loss_task2, loss_all, batch_task2 = self._calculate_total_loss(loss_task2, loss_all, loss_weight2)
            loss_task3, loss_all, batch_task3 = self._calculate_total_loss(loss_task3, loss_all, loss_weight3)
        
        labels_masked_task1, predictions_masked_task1, correct_count_task1, samples_count_task1 = self._get_predicted_and_count(prediction_task1, labels_task1)
        labels_masked_task2, predictions_masked_task2, correct_count_task2, samples_count_task2 = self._get_predicted_and_count(prediction_task2, labels_task2)
        labels_masked_task3, predictions_masked_task3, correct_count_task3, samples_count_task3 = self._get_predicted_and_count(prediction_task3, labels_task3)
        
        samples_count_all = samples_count_task1 + samples_count_task2 + samples_count_task3
        correct_count_all = correct_count_task1 + correct_count_task2 + correct_count_task3

        return TripleTaskBatchProcessOutput(
            loss_all=loss_all.item(),
            batch_all=batch_all,
            correct_count_all=correct_count_all,
            samples_count_all=samples_count_all,
            loss_task1=loss_task1.item(),
            batch_task1=batch_task1,
            correct_count_task1=correct_count_task1,
            samples_count_task1=samples_count_task1,
            loss_task2=loss_task2.item(),
            batch_task2=batch_task2,
            correct_count_task2=correct_count_task2,
            samples_count_task2=samples_count_task2,
            loss_task3=loss_task3.item(),
            batch_task3=batch_task3,
            correct_count_task3=correct_count_task3,
            samples_count_task3=samples_count_task3,
            labels_task1=labels_masked_task1,
            predictions_task1=predictions_masked_task1,
            labels_task2=labels_masked_task2,
            predictions_task2=predictions_masked_task2,
            labels_task3=labels_masked_task3,
            predictions_task3=predictions_masked_task3,
        )
    
    def _save_to_txt(self, content, filename):
        with open(filename, 'a') as f:
            print(content, file=f)

    def _calculate_accuracy(self, total_correct, total_samples):
        if total_samples == 0:
            return 0.0
        else:
            return total_correct / total_samples
    
    def _calculate_average_loss(self, total_loss, total_batches):
        if total_batches == 0:
            return 0.0
        else:
            return total_loss / total_batches

    def _process_data_loader(self, dataloader):
        self.model.eval()
        total_loss_all = 0.0
        total_batches_all = 0
        total_correct_all = 0
        total_samples_all = 0
        total_loss_task1 = 0.0
        total_batches_task1 = 0
        total_correct_task1 = 0
        total_samples_task1 = 0
        total_loss_task2 = 0.0
        total_batches_task2 = 0
        total_correct_task2 = 0
        total_samples_task2 = 0
        total_loss_task3 = 0.0
        total_batches_task3 = 0
        total_correct_task3 = 0
        total_samples_task3 = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_output = self._process_batch(batch)
                total_loss_all += batch_output.loss_all
                total_batches_all += batch_output.batch_all
                total_correct_all += batch_output.correct_count_all
                total_samples_all += batch_output.samples_count_all
                
                total_loss_task1 += batch_output.loss_task1
                total_batches_task1 += batch_output.batch_task1
                total_correct_task1 += batch_output.correct_count_task1
                total_samples_task1 += batch_output.samples_count_task1
                
                total_loss_task2 += batch_output.loss_task2
                total_batches_task2 += batch_output.batch_task2
                total_correct_task2 += batch_output.correct_count_task2
                total_samples_task2 += batch_output.samples_count_task2
                
                total_loss_task3 += batch_output.loss_task3
                total_batches_task3 += batch_output.batch_task3
                total_correct_task3 += batch_output.correct_count_task3
                total_samples_task3 += batch_output.samples_count_task3

        accuracy_all = self._calculate_accuracy(total_correct_all, total_samples_all)
        avg_loss_all = self._calculate_average_loss(total_loss_all, total_batches_all)
        
        accuracy_task1 = self._calculate_accuracy(total_correct_task1, total_samples_task1)
        avg_loss_task1 = self._calculate_average_loss(total_loss_task1, total_batches_task1)
        
        accuracy_task2 = self._calculate_accuracy(total_correct_task2, total_samples_task2)
        avg_loss_task2 = self._calculate_average_loss(total_loss_task2, total_batches_task2)
        
        accuracy_task3 = self._calculate_accuracy(total_correct_task3, total_samples_task3)
        avg_loss_task3 = self._calculate_average_loss(total_loss_task3, total_batches_task3)

        return TripleTaskEvaluatorOutput(
            avg_loss_all=avg_loss_all,
            accuracy_all=accuracy_all,
            avg_loss_task1=avg_loss_task1,
            accuracy_task1=accuracy_task1,
            avg_loss_task2=avg_loss_task2,
            accuracy_task2=accuracy_task2,
            avg_loss_task3=avg_loss_task3,
            accuracy_task3=accuracy_task3,
        )
        
    def _generate_output_info(self, loss, accuracy, task_name, result_text_path):
        output_info = f"Loss {task_name}: {loss:.4f}, Accuracy {task_name}: {accuracy:.4f}"
        print(output_info)
        if result_text_path is not None:
            self._save_to_txt(output_info, result_text_path)

    def _generate_text_array(self, input_integer):
        text_array = []
        for i in range(1, input_integer + 1):
            text_array.append(f"feature_{i}")
        return text_array
    
    def get_loss_and_accuracy(self, result_text_path=None):
        eval_output = self._process_data_loader(self.dataloader)
        
        self._generate_output_info(eval_output.avg_loss_all, eval_output.accuracy_all, "all", result_text_path)
        self._generate_output_info(eval_output.avg_loss_task1, eval_output.accuracy_task1, self.label_key_array[0], result_text_path)
        self._generate_output_info(eval_output.avg_loss_task2, eval_output.accuracy_task2, self.label_key_array[1], result_text_path)
        self._generate_output_info(eval_output.avg_loss_task3, eval_output.accuracy_task3, self.label_key_array[2], result_text_path)
        
    def get_labels_and_predictions(self, output_file_name):
        prediction_csv_file_task1 = f"{output_file_name}_prediction_{self.label_key_array[0]}.csv"
        prediction_csv_file_task2 = f"{output_file_name}_prediction_{self.label_key_array[1]}.csv"
        prediction_csv_file_task3 = f"{output_file_name}_prediction_{self.label_key_array[2]}.csv"
       
        with open(prediction_csv_file_task1, mode='w', newline='') as file1, open(prediction_csv_file_task2, mode='w', newline='') as file2, open(prediction_csv_file_task3, mode='w', newline='') as file3:
            writer1 = csv.writer(file1)
            writer1.writerow([f'True Labels-{self.label_key_array[0]}', f'Predicted Labels-{self.label_key_array[0]}'])
            writer2 = csv.writer(file2)
            writer2.writerow([f'True Labels-{self.label_key_array[1]}', f'Predicted Labels-{self.label_key_array[1]}'])
            writer3 = csv.writer(file3)
            writer3.writerow([f'True Labels-{self.label_key_array[2]}', f'Predicted Labels-{self.label_key_array[2]}'])
            
            for batch in self.dataloader:
                batch_output = self._process_batch(batch)
                
                labels_task1 = batch_output.labels_task1.cpu()
                predictions_task1 = batch_output.predictions_task1.cpu()
                labels_task2 = batch_output.labels_task2.cpu()
                predictions_task2 = batch_output.predictions_task2.cpu()
                labels_task3 = batch_output.labels_task3.cpu()
                predictions_task3 = batch_output.predictions_task3.cpu()
                
                if labels_task1.size()[0] == 1:
                    writer1.writerow([labels_task1.item(), predictions_task1.item()])
                if labels_task2.size()[0] == 1:
                    writer2.writerow([labels_task2.item(), predictions_task2.item()])
                if labels_task3.size()[0] == 1:
                    writer3.writerow([labels_task3.item(), predictions_task3.item()])