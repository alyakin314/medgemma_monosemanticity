from abc import ABC, abstractmethod
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class BaseMultimodalDataset(Dataset, ABC):
    """
    An abstract base class for handling image-text multimodal datasets.

    Parameters
    ----------
    root_dir : str
        Directory containing all journal subdirectories.
    processor : callable
        A callable processor that processes text and images into a batch.
    dataset_file: str, optional
        File within the directory containing the list of samples (usually json).
        Defaults to "dataset.json".
    data_key: str, optional
        If not None - returns entry[data_key].
        Defaults to None.

    Attributes
    ----------
    samples : list
        List of image-dict_of_strings pairs loaded from the dataset file.
    """

    def __init__(
        self,
        root_dir,
        # processor,
        dataset_file="dataset.json",
        data_key=None,
    ):
        self.root_dir = root_dir
        # self.processor = processor
        self.dataset_file = dataset_file
        self.data_key = data_key
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self):
        """
        Abstract method to load image-caption pairs.
        Must be implemented by subclasses.
        """
        pass

    def __len__(self):
        """
        Returns the total number of samples.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an image-entry pair by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image and its corresponding text as a dict.
        """
        image_path, entry = self.samples[idx]
        image = Image.open(image_path)
        return image, entry


class BaseMultimodalDataModule(ABC):
    """
    A base data module for handling data loading and batching for training,
    validation, and testing using multimodal data.

    Parameters
    ----------
    root_dir : str
        The directory containing all data.
    processor : callable
        A callable processor that processes text and images into a batch.
    data_key: str, optional
        If not None - treats entry as a dictionary and returns entry[data_key].
        Defaults to None.
    batch_size : int, optional
        Number of samples in each batch of data.
        Defaults to 32.
    val_size : float, optional
        The proportion of the dataset to use for validation.
        Defaults to 0.15.
    test_size : float, optional
        The proportion of the dataset to use for testing.
        Defaults to 0.10.
    max_tokens : int
        Maximum number of tokens for the tokenizer.
        Defaults to 256.
    num_workers : int, optional
        Number of subprocesses to use for data loading.
        Defaults to 0.
    distributed : bool, optional
        Whether to setup DistriubtedSamplers and pass them to DataLoaders.
        Defaults to False.

    Attributes
    ----------
    train_dataset : Dataset
        The subset of the dataset used for training.
    val_dataset : Dataset
        The subset of the dataset used for validation.
    test_dataset : Dataset
        The subset of the dataset used for testing.
    """

    def __init__(
        self,
        root_dir,
        processor,
        data_key=None,
        batch_size=32,
        val_size=0.10,
        test_size=0.10,
        max_tokens=256,
        num_workers=0,
        distributed=False,
        ift=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.processor = processor
        self.data_key = data_key
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.distributed = distributed
        self.ift = ift
        if self.ift:
            self.train_collate_fn = self.ift_train_collate_fn
        else:
            self.train_collate_fn = self.reg_train_collate_fn

    @abstractmethod
    def setup(self, stage=None):
        """
        Prepares the dataset by loading and splitting into training, validation,
        and test sets. Also creates DistributedSampler if FSDP is enabled.

        Parameters
        ----------
        stage : str, optional
            Stage to set up, either "fit", "validate", "test", or "predict".
            Defaults to None.
        """
        pass

    def setup_distributed(self, stage=None):
        if self.train_dataset is None:
            self.setup()
        self.distributed = True
        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True
        )
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            DataLoader instance for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=not self.distributed,
            sampler=self.train_sampler if self.distributed else None,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            DataLoader instance for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=False,
            sampler=self.val_sampler if self.distributed else None,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            DataLoader instance for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=False,
            sampler=self.test_sampler if self.distributed else None,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        """
        Returns a DataLoader for the test dataset. This is a wrapper for
        test_dataloader.

        Returns
        -------
        DataLoader
            DataLoader instance for the test dataset.
        """
        return self.test_dataloader()

    def reg_train_collate_fn(self, examples):
        """
        A custom collate function to process a batch of examples.

        Parameters
        ----------
        examples : list of tuples
            List of (image, ground_truth) tuples.

        Returns
        -------
        tuple
            A tuple containing input_ids, attention_mask, pixel_values,
            and labels.
        """
        images = []
        texts = []
        for example in examples:
            image, ground_truth = example
            images.append([image])
            prompt = self.processor.apply_chat_template(ground_truth)
            texts.append(prompt)

        print(images)
        print(texts)
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            add_special_tokens=False,
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # mask image soft-tokens so we don't compute loss on vision placeholders
        if hasattr(self.processor, "image_token_id"):
            labels[batch["input_ids"] == self.processor.image_token_id] = -100

        batch["labels"] = labels

        return batch

    def ift_train_collate_fn(self, examples):
        images = []
        texts = []
        for example in examples:
            image, ground_truth = example
            images.append([image])
            prompt = self.processor.apply_chat_template(ground_truth)
            texts.append(prompt)
        
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        # look up boundary tokens dynamically from the tokenizer
        tokenizer = self.processor.tokenizer
        start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        model_token_id = tokenizer.convert_tokens_to_ids("model")

        labels = torch.full_like(batch["input_ids"], fill_value=-100)

        # Iterate over each sequence in the batch
        for i in range(labels.shape[0]):
            in_model_turn = False
            # Iterate over each token in the sequence
            for j in range(labels.shape[1]):
                current_token = batch["input_ids"][i, j].item()

                if current_token == start_of_turn_id:
                    in_model_turn = False
                    if j + 1 < labels.shape[1]:
                        if batch["input_ids"][i, j + 1].item() == model_token_id:
                            in_model_turn = True

                if in_model_turn:
                    labels[i, j] = batch["input_ids"][i, j]

                if current_token == end_of_turn_id and in_model_turn:
                    in_model_turn = False

        # mask padding tokens
        labels[batch["input_ids"] == tokenizer.pad_token_id] = -100

        # mask image soft-tokens
        if hasattr(self.processor, "image_token_id"):
            labels[batch["input_ids"] == self.processor.image_token_id] = -100

        batch["labels"] = labels
        return batch

