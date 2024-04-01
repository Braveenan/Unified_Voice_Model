import torch
from torch.utils.data import DataLoader

class CustomEmbDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, pin_memory=True, drop_last=False):
        collate_fn = self.defined_collate
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory
        )

    def defined_collate(self, batch):
        sequences, content_indices, speaker_indices, emotion_indices = zip(*batch)

        input_seq = torch.stack(sequences)
        labels_task1 = torch.tensor(content_indices)
        labels_task2 = torch.tensor(speaker_indices)
        labels_task3 = torch.tensor(emotion_indices)

        return (input_seq, labels_task1, labels_task2, labels_task3)