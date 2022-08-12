from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from itertools import islice
from torch.utils.data import Dataset
import torchio as tio
from concurrent.futures import ThreadPoolExecutor
import random
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class MyQueue(Dataset):
    def __init__(
            self,
            subjects_dataset,
            max_length: int,
            samples_per_volume: int,
            sampler,
            num_workers: int = 0,
            shuffle_subjects: bool = True,
            shuffle_patches: bool = True,
            start_background: bool = True,
            verbose: bool = False,
            ):
        super().__init__()
        self.sampler = sampler
        self.num_workers = num_workers
        self.tpool = ThreadPoolExecutor(max_workers=num_workers)
        self.works = list([])
        self.subjects_dataset = subjects_dataset
        self.patches_list = list([])
        self.max_length = max_length
        self.samples_per_volume = samples_per_volume
        self.num_subjects = len(subjects_dataset)
        self.num_sampled_patches = 0
    
    def _fill(self):
        max_num_subjects_for_queue = self.max_length // self.samples_per_volume
        num_subjects_for_queue = min(self.num_subjects, max_num_subjects_for_queue)
        iterable = range(num_subjects_for_queue)
        for _ in iterable:
            self.tpool.submit(self.fetch_next)
    
    def fetch_next(self,):
        if len(self.patches_list) < self.samples_per_volume * 2:#self.max_length:
            subject = random.choice(self.subjects_dataset)
            iterable = self.sampler(subject)
            patches = list(islice(iterable, self.samples_per_volume))
            self.patches_list.extend(patches)
            #print('fetch_next')
        else:
            pass
            #print('dont need fetch_next', len(self.patches_list))
    def __len__(self):
        return len(self.patches_list)
    def __getitem__(self, _):
        if not self.patches_list:
            self.fetch_next()
        sample_patch = self.patches_list.pop()
        #print('fetch_next-1')
        self.works = [work for work in self.works if not work.done()]
        if len(self.works) <= self.num_workers:
            self.works.append(self.tpool.submit(self.fetch_next))
        
        #print('fetch_next-2')
        self.num_sampled_patches += 1
        return sample_patch