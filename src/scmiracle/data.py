import math
from pathlib import Path
import random
import numpy as np
import pandas as pd
import os
from typing import Iterator, Optional, TypeVar, Any, Dict

from scipy.io import mmread
from scipy.sparse import csr_matrix

import zipfile
import requests
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, Sampler

from .nn import transform_registry
from .utils import load_csv

_T_co = TypeVar('_T_co', covariant=True)


class BasicModDataset(Dataset):
    """
    Base class for modality data.
    """

    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int:
                Number of samples.
        """
        raise NotImplementedError("Subclasses should implement this method to return the dataset size.")

    def __subset__(self, indices) -> 'BasicModDataset':
        """
        Create a subset of the dataset based on the provided indices.

        Parameters:
            indices : list
                List of indices to include in the subset.

        Returns:
            BasicModDataset:
                A new dataset instance containing only the specified indices.
        """
        raise NotImplementedError("Subclasses should implement this method to create a subset of the dataset.")

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve the data item at the specified index (not implemented in base class).

        Parameters:
            idx : int
                The index of the data item.
        """
        raise NotImplementedError("Subclasses should implement this method to retrieve a data item.")


class VECDataset(BasicModDataset):
    """
    Dataset for vector-based data.

    Parameters:
        path : str
            Directory containing vector-based data files.
    """

    def __init__(self, path: str, real_dims: list = None, expected_dims: list = None,):
        super().__init__()
        self.root = path
        self.data_path = sorted(os.listdir(path))
        self.real_dims = real_dims
        self.expected_dims = expected_dims
        logging.debug(f'VECDataset initialized with {len(self.data_path)} files from {self.root}')
        logging.debug(f'Real dims: {self.real_dims}, Expected dims: {self.expected_dims}')

    def __len__(self) -> int:
        """
        Return the number of files in the vector dataset.

        Returns:
            int:
                Number of vector files in the dataset.
        """
        return len(self.data_path)

    def __subset__(self, indices):
        """
        Create a subset of the vector dataset based on the provided indices.

        Parameters:
            indices : list
                List of indices to include in the subset.

        Returns:
            VECDataset:
                A new VECDataset instance containing only the specified indices.
        """
        subset = VECDataset(self.root, real_dims=self.real_dims, expected_dims=self.expected_dims)
        subset.data_path = [self.data_path[i] for i in indices]
        return subset

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve the vector data at the specified index.

        Parameters:
            idx : int
                The index of the vector file.

        Returns:
            np.ndarray:
                The vector data as a NumPy array.
        """
        vector_data = np.array(
            load_csv(os.path.join(self.root, self.data_path[idx])), dtype=np.float32
        )[0]
        if self.real_dims is None:
            self.real_dims = [vector_data.shape[0]]
        if self.expected_dims is not None and sum(self.expected_dims) != sum(self.real_dims):
            # padding 0
            if len(self.real_dims) > 1:
                vector_data_padded = np.zeros((sum(self.expected_dims)), dtype=np.float32)
                start_idx = 0
                start_idx_real = 0
                for real_dim, expected_dim in zip(self.real_dims, self.expected_dims):
                    vector_data_padded[start_idx:start_idx + real_dim] = vector_data[start_idx_real:start_idx_real + real_dim]
                    start_idx += expected_dim
                    start_idx_real += real_dim
                vector_data = vector_data_padded
            else:
                vector_data_padded = np.zeros((self.expected_dims[0]), dtype=np.float32)
                vector_data_padded[:self.real_dims[0]] = vector_data
                vector_data = vector_data_padded
        
        return vector_data


class MTXDataset(BasicModDataset):
    """
    Dataset for mtx-based data.

    Parameters:
        mtx_file : str
            Path to the mtx file.
        real_dims : list, optional
            A list of integers representing the actual dimensions of the data.
            Used for padding if it differs from expected_dims.
        expected_dims : list, optional
            A list of integers representing the expected dimensions of the data
            after padding.
    """
    def __init__(self, mtx_file: str, real_dims: list = None, expected_dims: list = None):
        super().__init__()
        
        # Load the sparse matrix first
        if mtx_file.endswith('.mtx'):
            self._raw_data = csr_matrix(mmread(mtx_file))
        else:
            raise ValueError(f'Unsupported file format: {mtx_file}')
        
        self.real_dims = real_dims
        self.expected_dims = expected_dims
        logging.debug(f'MTXDataset initialized with raw shape {self._raw_data.shape} from {mtx_file}')
        logging.debug(f'Real dims: {self.real_dims}, Expected dims: {self.expected_dims}')

        # Convert to dense array and apply padding at initialization
        all_data = self._raw_data.toarray().astype(np.float32)
        logging.debug(f'MTXDataset raw data before padding: {all_data}')

        if self.real_dims is None:
            # If real_dims not set, infer from the actual data shape
            self.real_dims = [all_data.shape[1]]

        if self.expected_dims is not None and sum(self.expected_dims) != sum(self.real_dims):
            # Pad each row if dimensions don't match
            if len(self.real_dims) > 1:
                logging.debug('Padding MTXDataset with multiple dimensions.')
                # Handle multiple dimensions in a more complex padding scenario
                padded_data = np.zeros((all_data.shape[0], sum(self.expected_dims)), dtype=np.float32)
                start_idx_padded = 0
                start_idx_real = 0
                for real_dim, expected_dim in zip(self.real_dims, self.expected_dims):
                    padded_data[:, start_idx_padded:start_idx_padded + real_dim] = all_data[:, start_idx_real:start_idx_real + real_dim]
                    start_idx_padded += expected_dim
                    start_idx_real += real_dim
                self.data = padded_data
            else:
                # Simple padding for a single dimension
                padded_data = np.zeros((all_data.shape[0], self.expected_dims[0]), dtype=np.float32)
                padded_data[:, :self.real_dims[0]] = all_data[:, :self.real_dims[0]]
                self.data = padded_data
        else:
            self.data = all_data # No padding needed, just store the original dense data
        logging.debug(f'MTXDataset raw data after padding: {self.data}')
        logging.debug(f'MTXDataset final data shape after padding: {self.data.shape}')

    def get_all(self) -> np.ndarray:
        """
        Return all data in the dataset as a NumPy array.
        The data is already padded during initialization.

        Returns:
            np.ndarray:
                All data in the dataset as a NumPy array.
        """
        return self.data

    def __subset__(self, indices):
        """
        Create a subset of the mtx dataset based on the provided indices.

        Parameters:
            indices : list
                List of indices to include in the subset.

        Returns:
            MTXDataset:
                A new MTXDataset instance containing only the specified indices.
        """
        subset = MTXDataset.__new__(MTXDataset)
        subset.data = self.data[indices]
        subset.real_dims = self.real_dims
        subset.expected_dims = self.expected_dims
        return subset

    def __len__(self) -> int:
        """
        Return the number of rows in the matrix dataset.

        Returns:
            int:
                Number of rows in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve the matrix row at the specified index.
        The data is already padded during initialization.

        Parameters:
            idx : int
                The index of the matrix row.

        Returns:
            np.ndarray:
                The matrix row as a NumPy array.
        """
        return self.data[idx]
    

class CSVDataset(BasicModDataset):
    """
    Dataset for csv-based data.

    Parameters:
        csv_file : str
            Path to the CSV or compressed CSV file (csv.gz).
        real_dims : list, optional
            A list of integers representing the actual dimensions of the data.
            Used for padding if it differs from expected_dims.
        expected_dims : list, optional
            A list of integers representing the expected dimensions of the data
            after padding.
    """

    def __init__(self, csv_file: str, real_dims: list = None, expected_dims: list = None):
        super().__init__()
        
        # Load the data frame first
        if csv_file.endswith('.csv'):
            self._raw_data_frame = np.array(load_csv(csv_file))[1:, 1:].astype(np.float32)
        elif csv_file.endswith('.csv.gz'):
            self._raw_data_frame = pd.read_csv(csv_file, index_col=0).values.astype(np.float32)
        else:
            raise ValueError(f'Unsupported file format: {csv_file}')

        self.real_dims = real_dims
        self.expected_dims = expected_dims
        logging.debug(f'CSVDataset initialized with raw shape {self._raw_data_frame.shape} from {csv_file}')
        logging.debug(f'Real dims: {self.real_dims}, Expected dims: {self.expected_dims}')

        # Apply padding at initialization
        all_data = self._raw_data_frame

        if self.real_dims is None:
            # If real_dims not set, infer from the actual data shape
            self.real_dims = [all_data.shape[1]]

        if self.expected_dims is not None and sum(self.expected_dims) != sum(self.real_dims):
            # Pad each row if dimensions don't match
            if len(self.real_dims) > 1:
                # Handle multiple dimensions in a more complex padding scenario
                padded_data = np.zeros((all_data.shape[0], sum(self.expected_dims)), dtype=np.float32)
                for i in range(all_data.shape[0]):
                    current_row = all_data[i]
                    start_idx_padded = 0
                    start_idx_real = 0
                    for real_dim, expected_dim in zip(self.real_dims, self.expected_dims):
                        padded_data[i, start_idx_padded:start_idx_padded + real_dim] = current_row[start_idx_real:start_idx_real + real_dim]
                        start_idx_padded += expected_dim
                        start_idx_real += real_dim
                self.data_frame = padded_data
            else:
                # Simple padding for a single dimension
                padded_data = np.zeros((all_data.shape[0], self.expected_dims[0]), dtype=np.float32)
                padded_data[:, :self.real_dims[0]] = all_data[:, :self.real_dims[0]]
                self.data_frame = padded_data
        else:
            self.data_frame = all_data # No padding needed, just store the original data
        
        logging.debug(f'CSVDataset final data shape after padding: {self.data_frame.shape}')

    def __len__(self) -> int:
        """
        Return the number of rows in the matrix dataset.

        Returns:
            int:
                Number of rows in the dataset.
        """
        return len(self.data_frame)
    
    def __subset__(self, indices):
        """
        Create a subset of the csv dataset based on the provided indices.

        Parameters:
            indices : list
                List of indices to include in the subset.

        Returns:
            CSVDataset:
                A new CSVDataset instance containing only the specified indices.
        """
        subset = CSVDataset.__new__(CSVDataset)
        subset.data_frame = self.data_frame[indices]
        subset.real_dims = self.real_dims
        subset.expected_dims = self.expected_dims
        return subset

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve the matrix row at the specified index.
        The data is already padded during initialization.

        Parameters:
            idx : int
                The index of the matrix row.

        Returns:
            np.ndarray:
                The matrix row as a NumPy array.
        """
        return self.data_frame[idx]


modDataset_map = {'vec': VECDataset, 'csv': CSVDataset, 'mtx': MTXDataset}


class MultiModalDataset(Dataset):
    """
    A dataset class for handling multi-modal data with optional masking and transformations.

    Parameters:
        mod_dict : Dict[str, str]
            A dictionary mapping modality names to their respective file paths.
        mod_id_dict : Dict[str, int]
            A dictionary mapping modality names to their unique identifiers.
        file_type : Dict[str, str]
            A dictionary mapping modality names to their file types (e.g., 'vec', 'csv', 'mtx').
        mask_path : Optional[Dict[str, str]]
            A dictionary mapping modality names to their mask file paths, default is None.
        transform : Optional[Dict[str, str]]
            A dictionary specifying transformations to apply to each modality, default is None.

    Methods:
        __len__():
            Returns the size of the dataset.
        __getitem__(idx: int) -> Dict[str, Dict[str, Any]]:
            Retrieves the data at the given index across all modalities.
    """

    def __init__(
        self,
        mod_dict: Dict[str, str],
        mod_id_dict: Dict[str, int],
        file_type: Dict[str, str],
        mask_path: Optional[Dict[str, str]] = None,
        transform: Optional[Dict[str, str]] = None,
        real_dims: Optional[Dict[str, list]] = None,
        expected_dims: Optional[Dict[str, list]] = None,
    ):
        self.mod_dict = mod_dict
        self.mod_id_dict = mod_id_dict
        self.file_type = file_type
        self.mask_path = mask_path
        if expected_dims is not None:
            self.data = {
                modality: modDataset_map[file_type[modality]](path, expected_dims = expected_dims[modality], real_dims = real_dims[modality] if real_dims else None)
                for modality, path in self.mod_dict.items()
            }
        else:
            self.data = {
                modality: modDataset_map[file_type[modality]](path)
                for modality, path in self.mod_dict.items()
            }
        if expected_dims is None:
            self.mask = (
                {
                    modality: np.array(load_csv(mask_path[modality])[1][1:]).astype(np.float32)
                    for modality in mask_path
                }
                if mask_path
                else None
            )
        else:
            # padding mask by zero
            self.mask = {}
            if mask_path:
                for modality in mask_path:
                    raw_mask = np.array(load_csv(mask_path[modality])[1][1:]).astype(np.float32)
                    expected_dim = expected_dims[modality]
                    padded_mask = np.zeros((sum(expected_dim),), dtype=np.float32)
                    padded_mask[:len(raw_mask)] = raw_mask
                    self.mask[modality] = padded_mask
        self.transform = transform or {}
        self.size = len(next(iter(self.data.values())))  # Determine dataset size from the first modality

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int:
                The number of samples in the dataset.
        """
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves the data at the specified index across all modalities.

        Parameters:
            idx : int
                The index of the sample to retrieve.

        Returns:
            Dict[str, Dict[str, Any]]:
                A dictionary containing the following keys:
                    - 'x': Modality data at the given index, with optional transformations applied.
                    - 's': Modality IDs.
                    - 'e': Masking information, if available.
        """
        items = {'x': {}, 's': {}, 'e': {}}

        # Retrieve data for each modality
        for modality, dataset in self.data.items():
            # Get raw data
            items['x'][modality] = dataset[idx]

            # Apply transformation if specified
            if modality in self.transform:
                transform_fn = transform_registry.get(self.transform[modality])
                items['x'][modality] = transform_fn(items['x'][modality])

            # Store modality ID
            items['s'][modality] = np.array([self.mod_id_dict[modality]], dtype=np.int64)

        # Add joint ID
        items['s']['joint'] = np.array([self.mod_id_dict['joint']], dtype=np.int64)

        # Add masking information if available
        if self.mask:
            for modality, mask_data in self.mask.items():
                items['e'][modality] = mask_data

        return items

    def __subset__(self, indices) -> 'MultiModalDataset':
        """
        Create a subset of the multi-modal dataset based on the provided indices.

        Parameters:
            indices : list
                List of indices to include in the subset.
        Returns:
            MultiModalDataset:
                A new MultiModalDataset instance containing only the specified indices.
        """
        subset = MultiModalDataset.__new__(MultiModalDataset)
        subset.mod_dict = self.mod_dict
        subset.mod_id_dict = self.mod_id_dict
        subset.file_type = self.file_type
        subset.mask_path = self.mask_path
        subset.transform = self.transform
        subset.data = {
            modality: dataset.__subset__(indices)
            for modality, dataset in self.data.items()
        }
        if self.mask:
            subset.mask = {
                modality: self.mask[modality]
                for modality in self.mask
            }
        else:
            subset.mask = None
        subset.size = len(indices)
        return subset
    

class MultiBatchContinualLearningSampler(Sampler):
    """
    Custom sampler for multi-batch sampling across multiple datasets with continual learning logic.

    Parameters:
        data_source : Dataset
            Concatenated dataset containing replay and current data.
        shuffle : bool
            Whether to shuffle the samples within each dataset for replay, default is True.
        batch_size : int
            Number of samples per batch (for both current and replay), default is 1.
        n_current_datasets : int
            Number of datasets designated as 'current' datasets.
        n_replay_datasets : int
            Number of datasets designated as 'replay' datasets.
    """

    def __init__(
        self,
        data_source: Dataset,
        shuffle: bool = True,
        batch_size: int = 1,
        n_current_datasets: int = 0,
        n_replay_datasets: int = 0,
        n_max = 10000,
    ):
        super().__init__(data_source)
        if not hasattr(data_source, 'datasets') or not hasattr(data_source, 'cumulative_sizes'):
            raise ValueError('Data source must be a ConcatDataset or equivalent.')

        self.data = data_source
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_dataset = len(self.data.datasets) # Total number of sub-datasets
        self.n_current_datasets = n_current_datasets
        self.n_replay_datasets = n_replay_datasets
        self.n_max = n_max

        if self.n_current_datasets + self.n_replay_datasets != self.n_dataset:
            raise ValueError(
                f"Sum of n_current_datasets ({n_current_datasets}) and n_replay_datasets ({n_replay_datasets}) "
                f"must equal total number of datasets in data_source ({self.n_dataset})."
            )

        # Assuming replay datasets come first, then current datasets
        self.replay_indices_range = range(self.n_replay_datasets)
        self.current_indices_range = range(self.n_replay_datasets, self.n_dataset)

        # Calculate the maximum number of full batches any current dataset can provide
        # This determines the overall epoch length for current datasets
        self.max_current_batches_per_dataset = 0
        if self.n_current_datasets > 0:
            self.max_current_batches_per_dataset = max(
                math.ceil(len(self.data.datasets[i]) / self.batch_size)
                for i in self.current_indices_range
            )
        
        # Sampler for replay datasets (can be shuffled)
        self.ReplaySampler = (
            torch.utils.data.RandomSampler if shuffle else torch.utils.data.SequentialSampler
        )
        # Sampler for current datasets (always sequential to ensure full coverage)
        self.CurrentSampler = torch.utils.data.SequentialSampler

    def __len__(self) -> int:
        """
        Calculate the total number of samples across all sub-datasets based on the desired sampling strategy.
        Each "full cycle" involves iterating through all current datasets once,
        potentially interleaved with replay batches.
        The length is primarily driven by ensuring all 'current' data is processed.

        Returns:
            int:
                The total number of samples.
        """
        # A "logical epoch" means we've iterated through each current dataset
        # `max_current_batches_per_dataset` ensures that even the largest current dataset is fully covered.
        # In each 'step', we aim to take one batch from a current dataset and one from a replay dataset (if available).
        
        # If there are current datasets, the total iterations are driven by them.
        # Each full pass through all current datasets provides `self.n_current_datasets` batches from current.
        # If replay exists, we match that with `self.n_current_datasets` batches from replay.
        
        total_batches_per_full_current_pass = self.n_current_datasets
        if self.n_replay_datasets > 0:
            total_batches_per_full_current_pass += self.n_current_datasets # For replay matching current
        
        # The total number of cycles is determined by how many times we need to go through
        # the current datasets to cover the largest one.
        num_cycles = min(self.max_current_batches_per_dataset, self.n_max)
        
        return num_cycles * total_batches_per_full_current_pass * self.batch_size


    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the dataset indices in a multi-batch continual learning manner,
        alternating between current and replay datasets.

        Returns:
            Iterator[int]:
                An iterator over sampled indices.
        """
        # --- Initialize Samplers and Iterators ---
        replay_samplers = [
            self.ReplaySampler(self.data.datasets[idx]) for idx in self.replay_indices_range
        ]
        replay_iterators = [iter(s) for s in replay_samplers]

        current_samplers = [
            self.CurrentSampler(self.data.datasets[idx]) for idx in self.current_indices_range
        ]
        current_iterators = [iter(s) for s in current_samplers]
        
        # Cumulative sizes for offset indexing (all datasets)
        push_index_val = [0] + self.data.cumulative_sizes[:-1]

        all_indices = []
        
        # We need to manage the state of current datasets, as they will be cycled through.
        current_dataset_ids = list(self.current_indices_range)
        
        # We will loop enough times to ensure all current data is processed at least once.
        # This loop will represent the "main" progression through the current datasets.
        
        current_dataset_step_counters = {
            idx: 0 for idx in self.current_indices_range
        }
        
        # This will run until all current datasets have yielded their maximum batches
        # based on `max_current_batches_per_dataset`.
        
        # The overall iteration should continue as long as there's any current data left to process
        # up to the calculated max_current_batches_per_dataset for the largest current dataset.
        
        # To ensure fair interleaving and cover all current data:
        # We will iterate `max_current_batches_per_dataset` times.
        # In each of these main loops, we will attempt to get one batch from each current dataset (in shuffled order)
        # and for each such current batch, we will also get a replay batch.

        for _ in range(self.max_current_batches_per_dataset):
            random.shuffle(current_dataset_ids) # Shuffle order of current datasets for interleaving
            
            for i_global_current in current_dataset_ids:
                # --- Sample from a Current Dataset ---
                i_local_current = i_global_current - self.n_replay_datasets
                current_s = current_iterators[i_local_current]
                
                current_batch_indices = []
                for _ in range(self.batch_size):
                    try:
                        current_batch_indices.append(next(current_s) + push_index_val[i_global_current])
                    except StopIteration:
                        # If a current dataset is exhausted, restart its sampler.
                        # This handles cases where current datasets have different sizes.
                        current_iterators[i_local_current] = iter(current_samplers[i_local_current])
                        current_s = current_iterators[i_local_current]
                        current_batch_indices.append(next(current_s) + push_index_val[i_global_current])
                
                all_indices.extend(current_batch_indices)

                # --- Sample from a Replay Dataset (if available) ---
                if self.n_replay_datasets > 0:
                    i_global_replay = random.choice(self.replay_indices_range)
                    replay_s = replay_iterators[i_global_replay]

                    replay_batch_indices = []
                    for _ in range(self.batch_size):
                        try:
                            replay_batch_indices.append(next(replay_s) + push_index_val[i_global_replay])
                        except StopIteration:
                            # Restart replay sampler if exhausted
                            replay_iterators[i_global_replay] = iter(replay_samplers[i_global_replay])
                            replay_s = replay_iterators[i_global_replay]
                            replay_batch_indices.append(next(replay_s) + push_index_val[i_global_replay])
                    all_indices.extend(replay_batch_indices)
        
        return iter(all_indices)

    

class MultiBatchSampler(Sampler):
    """
    Custom sampler for multi-batch sampling across multiple datasets.

    Parameters:
        data_source : Dataset
            Dataset.
        shuffle : bool
            Whether to shuffle the samples within each dataset, default is True.
        batch_size : int
            Number of samples per batch, default is 1.
        n_max : int
            Maximum number of samples to draw from each dataset, default is 10000.
    """

    def __init__(
        self,
        data_source: Dataset,
        shuffle: bool = True,
        batch_size: int = 1,
        n_max: int = 10000,
    ):
        super().__init__(data_source)
        if not hasattr(data_source, 'datasets') or not hasattr(data_source, 'cumulative_sizes'):
            raise ValueError('Data source must be a ConcatDataset or equivalent.')

        self.data = data_source
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_dataset = len(self.data.datasets)
        self.n_max = min(max(len(d) for d in self.data.datasets), n_max)
        self.Sampler = (
            torch.utils.data.RandomSampler if shuffle else torch.utils.data.SequentialSampler
        )

    def __len__(self) -> int:
        """
        Calculate the total number of samples across all sub-datasets.

        Returns:
            int:
                The total number of samples.
        """
        return math.ceil(self.n_max / self.batch_size) * self.batch_size * self.n_dataset

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the dataset indices in a multi-batch sampling manner.

        Returns:
            Iterator[int]:
                An iterator over sampled indices.
        """
        # Number of iterations per dataset
        n_iter = math.ceil(self.n_max / self.batch_size)

        # Create individual samplers and iterators for each dataset
        sampler_indv = [
            self.Sampler(self.data.datasets[idx]) for idx in range(self.n_dataset)
        ]
        sampler_iter_indv = [iter(s) for s in sampler_indv]

        # Cumulative sizes for offset indexing
        push_index_val = [0] + self.data.cumulative_sizes[:-1]
        idx_dataset = list(range(self.n_dataset))

        indices = []
        for _ in range(n_iter):
            # Shuffle dataset order if required
            if self.shuffle:
                random.shuffle(idx_dataset)

            for i in idx_dataset:
                s = sampler_iter_indv[i]
                indices_indv = []
                for _ in range(self.batch_size):
                    try:
                        indices_indv.append(next(s) + push_index_val[i])
                    except StopIteration:
                        # Restart sampler iterator if exhausted
                        sampler_iter_indv[i] = iter(sampler_indv[i])
                        s = sampler_iter_indv[i]
                        indices_indv.append(next(s) + push_index_val[i])
                indices.extend(indices_indv)

        return iter(indices)


class MyDistributedSampler(DistributedSampler):
    """
    A custom distributed sampler for datasets split across multiple replicas.

    Parameters:
        dataset : Dataset
            The dataset to sample from.
        num_replicas : Optional[int]
            Number of replicas in the distributed setup, default is determined by `torch.distributed`.
        rank : Optional[int]
            The rank of the current process, default is determined by `torch.distributed`.
        shuffle : bool
            Whether to shuffle the data, default is True.
        seed : int
            Random seed for shuffling, default is 0.
        batch_size : int
            Number of samples per batch, default is 256.
        n_max : int
            Maximum number of samples per dataset, default is 10000.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        batch_size: int = 256,
        n_max: int = 10000,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f'Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]'
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.n_dataset = len(self.dataset.datasets)
        self.n_sample = [len(d) // num_replicas for d in self.dataset.datasets]
        self.batch_size = batch_size
        self.n_max = n_max

        # Cumulative dataset sizes for indexing
        self.push_index_val = [0] + self.dataset.cumulative_sizes
        self.all_indices = []
        self.all_length = []

        # Generate indices for each dataset
        for idx in range(self.n_dataset):
            indices = list(
                range(
                    self.rank + self.push_index_val[idx],
                    self.push_index_val[idx + 1],
                    self.num_replicas,
                )
            )
            self.all_indices.append(indices)
            self.all_length.append(len(indices))

    def __iter__(self) -> Iterator[_T_co]:
        """
        Iterate over the distributed dataset, ensuring balanced sampling across replicas.

        Returns:
            Iterator:
                Iterator over indices for the current replica.
        """
        sampler_indv = []
        sampler_iter_indv = []
        n_sample_by_dataset = []

        # Prepare samplers for each dataset
        for idx in range(self.n_dataset):
            indices = self.all_indices[idx]
            if self.shuffle:
                random.shuffle(indices)
            indices = indices[: self.n_max]
            sampler_indv.append(indices)
            sampler_iter_indv.append(iter(indices))
            n_sample_by_dataset.append(len(indices))

        n_iter = math.ceil(max(n_sample_by_dataset) / self.batch_size) * self.n_dataset

        idx_dataset = list(range(self.n_dataset))
        indices = []

        # Main sampling loop
        for _ in range(n_iter):
            random.shuffle(idx_dataset)  # Shuffle dataset order
            for i in idx_dataset:
                s = sampler_iter_indv[i]
                order_indv = []
                for _ in range(self.batch_size):
                    try:
                        order_indv.append(next(s))
                    except StopIteration:
                        sampler_iter_indv[i] = iter(sampler_indv[i])
                        s = sampler_iter_indv[i]
                        order_indv.append(next(s))
                indices.extend(order_indv)

        return iter(indices)

    def __len__(self) -> int:
        """
        Calculate the number of samples in the sampler.

        Returns:
            int:
                Number of samples across all datasets.
        """
        return sum(self.all_length)
        # max_samples = min(max(self.all_length), self.n_max)
        # return math.ceil(max_samples / self.batch_size) * self.n_dataset * self.batch_size

def download_file(url: str, dest_path: str):
    """Helper function to download a file from a URL with progress display.
    
    Parameters:
        url : str
            URL for data.
        dest_path : str
            Path to save.
    """
    try:
        # Send HTTP GET request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the total size of the file from headers
        total_size = int(response.headers.get('Content-Length', 0))
        
        # Open the destination file in write-binary mode
        with open(dest_path, 'wb') as file:
            # Use tqdm to display download progress
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {dest_path.name}') as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))  # Update progress bar with the downloaded chunk size
        logging.info(f'Downloaded: {url} to {dest_path}')

    except requests.exceptions.RequestException as e:
        logging.error(f'Error downloading {url}: {e}')
        raise

def unzip_file(zip_path: str, extract_to: str):
    """Helper function to unzip a file.

    Parameters:
        zip_path : str
            Path of zip file.
        extract_to : str
            Path to save.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f'Unzipped: {zip_path} to {extract_to}')
    except zipfile.BadZipFile as e:
        logging.error(f'Error unzipping {zip_path}: {e}')
        raise

def download_models(name: str, des: str = './'):
    """
    Downloads the specified model.

    Parameters:
        name : str
            Name of the model to download (e.g., 'wnn_mosaic_8batch_mtx').
        des : str
            Destination path to save the model (default is the current directory).
    """
    # Set up the destination path
    des_path = Path(des) / 'saved_models'
    des_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    urls_dict = {
        'wnn_full_8batch_mtx' : [('https://pub-cfde59ed245349228f47377c9ae32dd3.r2.dev/wnn_full_8batch_mtx.pt', des_path / 'wnn_full_8batch_mtx.pt')],
        'wnn_mosaic_8batch_mtx' : [('https://pub-cfde59ed245349228f47377c9ae32dd3.r2.dev/wnn_mosaic_8batch_mtx.pt', des_path / 'wnn_mosaic_8batch_mtx.pt')],
        'teadog_mosaic_mtx' : [('https://pub-cfde59ed245349228f47377c9ae32dd3.r2.dev/teadog_mosaic_mtx.pt', des_path / 'teadog_mosaic_mtx.pt')],
    
    }

    if name in urls_dict:
        try:
            # Download and extract the TEADOG mosaic dataset
            urls = urls_dict[name]
            for url, file_path in urls:
                logging.info(f'Downloading from {url}.')
                download_file(url, file_path)
                if file_path.suffix == '.zip':
                    unzip_file(file_path, des_path)
                    os.remove(file_path)
        except Exception as e:
            logging.error(f'An error occurred while downloading the dataset: {e}')
            raise

    else:
        logging.error(f'Dataset "{name}" is not recognized.')
        raise ValueError(f'Dataset "{name}" not supported.')

def download_data(name: str, des: str = './'):
    """
    Downloads the specified dataset and extracts it.

    Parameters:
        name : str
            Name of the dataset to download (e.g., 'teadog_mosaic_4k').
        des : str
            Destination path to save the dataset (default is the current directory).
    """
    # Set up the destination path
    des_path = Path(des) / 'dataset'
    des_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    urls_dict = {
        'teadog_mosaic_4k' : [('https://drive.usercontent.google.com/download?id=1MQtg5CHV3KDsmbRowiNnggKImYazBpOi&export=download&authuser=0&confirm=t&uuid=840e8dbf-6a9b-407f-89fe-cc5c82debc8a&at=APvzH3omA-S-4W1YkjAlCvyM6EuX:1733823042031', des_path / 'teadog_mosaic_4k.zip')],
        'wnn_mosaic_3batch' : [('https://drive.usercontent.google.com/download?id=11a62mlJ4tbqPMM7y6iF9XfMxeWMFqc-7&export=download&authuser=0&confirm=t&uuid=f6efdc19-ba0b-448a-bfa1-ab65a9784bee&at=APvzH3rBWhgaiST18uqbTjSu6uo4:1734661218069', des_path / 'wnn_mosaic_3batch.zip')],
        'wnn_full_3batch' : [('https://drive.usercontent.google.com/download?id=1W3ZkU8TWzlPcCuqlGvptfH_PnHjvWI4u&export=download&authuser=0&confirm=t&uuid=015fddd9-a789-4bc7-8fda-3f4ef202811a&at=APvzH3rhfWzjXrlKJedDEBGzhsXm:1734661020282', des_path / 'wnn_full_3batch.zip')],
        'wnn_full_8batch' : [('https://drive.usercontent.google.com/download?id=1kzlSd6iAM2UHifvlzu0OYbpq_MLPomrx&export=download&authuser=0&confirm=t&uuid=79c4ce32-18ca-4ba3-bbbd-e1c955ab1064&at=APvzH3q3nmmKLDSI1SNtF1CGNbnn:1734661120552', des_path / 'wnn_full_8batch.zip')],
        'wnn_full_8batch_mtx' : [('https://pub-cfde59ed245349228f47377c9ae32dd3.r2.dev/wnn_full_8batch_mtx.zip', des_path / 'wnn_full_8batch_mtx.zip')],
        'wnn_mosaic_8batch_mtx' : [('https://pub-cfde59ed245349228f47377c9ae32dd3.r2.dev/wnn_mosaic_8batch_mtx.zip', des_path / 'wnn_mosaic_8batch_mtx.zip')],
        'teadog_mosaic_mtx' : [('https://pub-cfde59ed245349228f47377c9ae32dd3.r2.dev/teadog_mosaic_mtx.zip', des_path / 'teadog_mosaic_mtx.zip')],
    }

    if name in urls_dict:
        try:
            # Download and extract the TEADOG mosaic dataset
            urls = urls_dict[name]
            for url, file_path in urls:
                logging.info(f'Downloading from {url}.')
                download_file(url, file_path)
                if file_path.suffix == '.zip':
                    unzip_file(file_path, des_path)
                    os.remove(file_path)
        except Exception as e:
            logging.error(f'An error occurred while downloading the dataset: {e}')
            raise

    else:
        logging.error(f'Dataset "{name}" is not recognized.')
        raise ValueError(f'Dataset "{name}" not supported.')

def download_script(name: str, des: str = './'):
    """
    Downloads the specified script.

    Parameters:
        name : str
            Name of the script to download (e.g., 'wnn_bimodal.R').
        des : str
            Destination path to save the script (default is the current directory).
    """
    
    des_path = Path(des)
    des_path.mkdir(parents=True, exist_ok=True)
    url_dict = {
        'wnn_bimodal.R': 'https://raw.githubusercontent.com/labomics/midas/main/docs/source/tutorials/basics/wnn_bimodal.R',
        'wnn_trimodal.R': 'https://raw.githubusercontent.com/labomics/midas/main/docs/source/tutorials/basics/wnn_trimodal.R',
    }
    
    if name in url_dict:
        url = url_dict[name]
        file_path = des_path / name
        if not file_path.exists():
            logging.warning(f"'{file_path}' not found. Downloading...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                file_path.write_text(response.text, encoding='utf-8')
                logging.info(f"Successfully downloaded '{file_path}'.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error downloading the file: {e}")
                raise
        else:
            logging.warning(f"'{file_path}' already exists. Skipping download.")           
    else:
        logging.error(f'Script "{name}" is not recognized.')
        raise ValueError(f'Script "{name}" not supported.')