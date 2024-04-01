import os
import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.ticker import LogLocator

class TripleTaskTrainerOutput:
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
    def __init__(self, loss_all=None, batch_all=None, correct_count_all=None, samples_count_all=None, loss_task1=None, batch_task1=None, correct_count_task1=None, samples_count_task1=None, loss_task2=None, batch_task2=None, correct_count_task2=None, samples_count_task2=None, loss_task3=None, batch_task3=None, correct_count_task3=None, samples_count_task3=None):
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
   
    
class TripleTasksModelTrainer:
    def __init__(self, model, optimizer_parameters, scheduler_parameters, device, label_key_array, num_epochs, saved_checkpoint_count, l1_lambda, l2_lambda):
        self.model = model
        self.device = device
        self.label_key_array = label_key_array
        self.num_epochs = num_epochs
        self.saved_checkpoint_count = saved_checkpoint_count
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.optimizer = AdamW(model.parameters(), lr=optimizer_parameters['learning_rate'], weight_decay=optimizer_parameters['weight_decay'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=scheduler_parameters['patience'], factor=scheduler_parameters['factor'], verbose=True)
        
        self.test_accuracy_all_threshold = 0.0
        self.test_accuracy_task1_threshold = 0.0
        self.test_accuracy_task2_threshold = 0.0
        self.test_accuracy_task3_threshold = 0.0
        
        self.train_losses_all = []
        self.train_accuracies_all = []
        self.test_losses_all = []
        self.test_accuracies_all = []

        self.train_losses_task1 = []
        self.train_accuracies_task1 = []
        self.test_losses_task1 = []
        self.test_accuracies_task1 = []

        self.train_losses_task2 = []
        self.train_accuracies_task2 = []
        self.test_losses_task2 = []
        self.test_accuracies_task2 = []

        self.train_losses_task3 = []
        self.train_accuracies_task3 = []
        self.test_losses_task3 = []
        self.test_accuracies_task3 = []

        self.learning_rate_array = []
        
    def _get_count_batch(self, prediction, labels):
        mask = labels != -1
        labels_masked = labels[mask]
        prediction_masked = prediction[mask]
        correct_count = (prediction_masked == labels_masked).sum().item()
        samples_count = labels_masked.size(0)
        return correct_count, samples_count
    
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
        data_count = None
        if labels is not None:
            mask = labels != -1
            labels_masked = labels[mask]
            if labels_masked.size(0) != 0:
                logits_masked = logits[mask, :]
                loss = loss_fn(logits_masked, labels_masked)

            data_count = labels_masked.size(0)
        return loss, data_count
        
    def _process_batch(self, batch, train_mode=True):
        input_seq = batch[0].to(self.device)
        labels_task1 = batch[1].to(self.device)
        labels_task2 = batch[2].to(self.device)
        labels_task3 = batch[3].to(self.device)

        self.optimizer.zero_grad()

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

        loss_task1, data_count1 = self._calculate_batch_loss(logits_task1, labels_task1, self.loss_fn)
        loss_task2, data_count2 = self._calculate_batch_loss(logits_task2, labels_task2, self.loss_fn)
        loss_task3, data_count3 = self._calculate_batch_loss(logits_task3, labels_task3, self.loss_fn)
    
        loss_task1, loss_all, batch_task1 = self._calculate_total_loss(loss_task1, loss_all, loss_weight1)
        loss_task2, loss_all, batch_task2 = self._calculate_total_loss(loss_task2, loss_all, loss_weight2)
        loss_task3, loss_all, batch_task3 = self._calculate_total_loss(loss_task3, loss_all, loss_weight3)
        
        # Get the model parameters
        model_params = [param for param in self.model.parameters()]
        
        # Calculate L1 regularization term
        l1_regularization = torch.tensor(0., device=self.device)
        for param in model_params:
            l1_regularization += torch.sum(torch.abs(param))

        # Add L1 regularization term to the total loss
        loss_all += self.l1_lambda * l1_regularization

        # Calculate L2 regularization term
        l2_regularization = torch.tensor(0., device=self.device)
        for param in model_params:
            l2_regularization += torch.sum(torch.square(param))

        # Add L2 regularization term to the total loss
        loss_all += self.l2_lambda * l2_regularization
            
        if train_mode:
            loss_all.backward()
            self.optimizer.step()
            
        correct_count_task1, samples_count_task1 = self._get_count_batch(prediction_task1, labels_task1)
        correct_count_task2, samples_count_task2 = self._get_count_batch(prediction_task2, labels_task2)
        correct_count_task3, samples_count_task3 = self._get_count_batch(prediction_task3, labels_task3)
        
        samples_count_all = samples_count_task1 + samples_count_task2 + samples_count_task3
        correct_count_all = correct_count_task1 + correct_count_task2 + correct_count_task3

        if self.data_count_path is not None:
            operation = "train" if train_mode else "val"
            data_count_info = f"{operation} {data_count1} {data_count2} {data_count3}"
            self._save_to_txt(data_count_info, self.data_count_path)

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
        )
    
    def _process_data_loader(self, data_loader, train_mode=True):
        self.model.train() if train_mode else self.model.eval()
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

        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                batch_output = self._process_batch(batch, train_mode)
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

        return TripleTaskTrainerOutput(
            avg_loss_all=avg_loss_all,
            accuracy_all=accuracy_all,
            avg_loss_task1=avg_loss_task1,
            accuracy_task1=accuracy_task1,
            avg_loss_task2=avg_loss_task2,
            accuracy_task2=accuracy_task2,
            avg_loss_task3=avg_loss_task3,
            accuracy_task3=accuracy_task3,
        )
    
    def _train_epoch(self, data_loader):
        train_output = self._process_data_loader(data_loader, train_mode=True)

        return train_output

    def _test_epoch(self, data_loader):
        test_output = self._process_data_loader(data_loader, train_mode=False)

        return test_output
    
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

    def _save_to_txt(self, content, filename):
        with open(filename, 'a') as f:
            print(content, file=f)

    def _plot_metric(self, num_epochs, train_data, test_data, plot_path, plot_title, metric_name):
        # Create a figure
        plt.figure(figsize=(16, 8))
    
        # Plot Train and Test data
        plt.plot(range(1, num_epochs + 1), train_data, label=f'Train {metric_name}', marker='o')
        plt.plot(range(1, num_epochs + 1), test_data, label=f'Test {metric_name}', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to discrete values
    
        # Set title and save the figure
        plt.title(plot_title)
        plt.tight_layout()
    
        # Modify the plot_path based on metric_name
        modified_plot_path = plot_path.replace(".png", f"_{metric_name}.png")
    
        plt.savefig(modified_plot_path, transparent=True)
        plt.close()
        plt.clf()

    def _plot_lr(self, num_epochs, lr_array, plot_path, plot_title):
        # Create a figure
        plt.figure(figsize=(16, 8))
    
        # Plot Train and Test data
        plt.plot(range(1, num_epochs + 1), lr_array, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.legend()
        plt.xticks(range(1, num_epochs + 1))  # Set x-axis ticks to discrete values

        plt.yscale('log')

        plt.gca().yaxis.set_major_locator(LogLocator())
    
        # Set title and save the figure
        plt.title(plot_title)
        plt.tight_layout()
    
        # Modify the plot_path based on metric_name
        modified_plot_path = plot_path.replace(".png", f"_lr.png")
    
        plt.savefig(modified_plot_path, transparent=True)
        plt.close()
        plt.clf()
    
    def _plot_function(self, num_epochs, train_accuracies, test_accuracies, train_losses, test_losses, plot_path, plot_title):
        self._plot_metric(num_epochs, train_accuracies, test_accuracies, plot_path, plot_title, "Accuracy")
        self._plot_metric(num_epochs, train_losses, test_losses, plot_path, plot_title, "Loss")
        
    def _save_model_checkpoint(self, model_checkpoint_path, epoch, test_accuracy, saved_checkpoint_count):
        model_epoch_checkpoint_path = model_checkpoint_path.replace(".pth", f"_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), model_epoch_checkpoint_path)
        
        if epoch > saved_checkpoint_count:
            model_epoch_checkpoint_to_remove = model_checkpoint_path.replace(".pth", f"_epoch{epoch-saved_checkpoint_count}.pth")
            if os.path.exists(model_epoch_checkpoint_to_remove):
                os.remove(model_epoch_checkpoint_to_remove)

        if test_accuracy["all"] > self.test_accuracy_all_threshold:
            self.test_accuracy_all_threshold = test_accuracy["all"]
            self.test_accuracy_task1_threshold = test_accuracy["task1"]
            self.test_accuracy_task2_threshold = test_accuracy["task2"]
            self.test_accuracy_task3_threshold = test_accuracy["task3"]
            model_best_checkpoint_path = model_checkpoint_path.replace(".pth", f"_best.pth")
            torch.save(self.model.state_dict(), model_best_checkpoint_path)
        
    def _generate_epoch_info(self, epoch, num_epochs, train_loss, train_accuracy, test_loss, test_accuracy, task_name, result_text_path):
        epoch_info = f"Epoch {epoch+1}/{num_epochs} - Train Loss {task_name}: {train_loss:.4f}, Train Accuracy {task_name}: {train_accuracy:.4f}, Test Loss {task_name}: {test_loss:.4f}, Test Accuracy {task_name}: {test_accuracy:.4f}"
        print(epoch_info)
        if result_text_path is not None:
            self._save_to_txt(epoch_info, result_text_path)
        
    def train(self):
        for epoch in range(self.num_epochs):
            train_output = self._train_epoch(self.train_dataloader)
            test_output = self._test_epoch(self.test_dataloader)

            train_loss_all, train_accuracy_all = train_output.avg_loss_all, train_output.accuracy_all
            test_loss_all, test_accuracy_all = test_output.avg_loss_all, test_output.accuracy_all
            
            train_loss_task1, train_accuracy_task1 = train_output.avg_loss_task1, train_output.accuracy_task1
            test_loss_task1, test_accuracy_task1 = test_output.avg_loss_task1, test_output.accuracy_task1
            
            train_loss_task2, train_accuracy_task2 = train_output.avg_loss_task2, train_output.accuracy_task2
            test_loss_task2, test_accuracy_task2 = test_output.avg_loss_task2, test_output.accuracy_task2
            
            train_loss_task3, train_accuracy_task3 = train_output.avg_loss_task3, train_output.accuracy_task3
            test_loss_task3, test_accuracy_task3 = test_output.avg_loss_task3, test_output.accuracy_task3

            self.train_losses_all.append(train_loss_all)
            self.train_accuracies_all.append(train_accuracy_all)
            self.test_losses_all.append(test_loss_all)
            self.test_accuracies_all.append(test_accuracy_all)
            
            self.train_losses_task1.append(train_loss_task1)
            self.train_accuracies_task1.append(train_accuracy_task1)
            self.test_losses_task1.append(test_loss_task1)
            self.test_accuracies_task1.append(test_accuracy_task1)
            
            self.train_losses_task2.append(train_loss_task2)
            self.train_accuracies_task2.append(train_accuracy_task2)
            self.test_losses_task2.append(test_loss_task2)
            self.test_accuracies_task2.append(test_accuracy_task2)
            
            self.train_losses_task3.append(train_loss_task3)
            self.train_accuracies_task3.append(train_accuracy_task3)
            self.test_losses_task3.append(test_loss_task3)
            self.test_accuracies_task3.append(test_accuracy_task3)
            
            self._generate_epoch_info(epoch, self.num_epochs, train_loss_all, train_accuracy_all, test_loss_all, test_accuracy_all, "all", self.result_text_path_all)               
            self._generate_epoch_info(epoch, self.num_epochs, train_loss_task1, train_accuracy_task1, test_loss_task1, test_accuracy_task1, self.label_key_array[0], self.result_text_path_task1)             
            self._generate_epoch_info(epoch, self.num_epochs, train_loss_task2, train_accuracy_task2, test_loss_task2, test_accuracy_task2, self.label_key_array[1], self.result_text_path_task2)           
            self._generate_epoch_info(epoch, self.num_epochs, train_loss_task3, train_accuracy_task3, test_loss_task3, test_accuracy_task3, self.label_key_array[2], self.result_text_path_task3)

            test_accuracy = {
                "all": test_accuracy_all,
                "task1": test_accuracy_task1,
                "task2": test_accuracy_task2,
                "task3": test_accuracy_task3
            }
                
            self._save_model_checkpoint(self.model_checkpoint_path, epoch+1, test_accuracy, self.saved_checkpoint_count)
            self.scheduler.step(test_loss_all)
            learning_rate_value = self.optimizer.param_groups[0]["lr"]
            self.learning_rate_array.append(learning_rate_value)

        best_accuracy_text_all = f"Best model accuracy - all: {self.test_accuracy_all_threshold}"
        print(best_accuracy_text_all)
        self._save_to_txt(best_accuracy_text_all, self.result_text_path_all)

        best_accuracy_text_task1 = f"Best model accuracy - {self.label_key_array[0]}: {self.test_accuracy_task1_threshold}"
        print(best_accuracy_text_task1)
        self._save_to_txt(best_accuracy_text_task1, self.result_text_path_task1)

        best_accuracy_text_task2 = f"Best model accuracy - {self.label_key_array[1]}: {self.test_accuracy_task2_threshold}"
        print(best_accuracy_text_task2)
        self._save_to_txt(best_accuracy_text_task2, self.result_text_path_task2)

        best_accuracy_text_task3 = f"Best model accuracy - {self.label_key_array[2]}: {self.test_accuracy_task3_threshold}"
        print(best_accuracy_text_task3)
        self._save_to_txt(best_accuracy_text_task3, self.result_text_path_task3)
            
    def plot_metrics(self):
        self._plot_function(self.num_epochs, self.train_accuracies_all, self.test_accuracies_all, self.train_losses_all, self.test_losses_all, self.result_plot_path_all, self.plot_title_all)

        self._plot_function(self.num_epochs, self.train_accuracies_task1, self.test_accuracies_task1, self.train_losses_task1, self.test_losses_task1, self.result_plot_path_task1, self.plot_title_task1)

        self._plot_function(self.num_epochs, self.train_accuracies_task2, self.test_accuracies_task2, self.train_losses_task2, self.test_losses_task2, self.result_plot_path_task2, self.plot_title_task2)

        self._plot_function(self.num_epochs, self.train_accuracies_task3, self.test_accuracies_task3, self.train_losses_task3, self.test_losses_task3, self.result_plot_path_task3, self.plot_title_task3)

        self. _plot_lr(self.num_epochs, self.learning_rate_array, self.result_plot_path_all, self.plot_title_all)
    
    
