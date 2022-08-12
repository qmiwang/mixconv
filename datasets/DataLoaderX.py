from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from itertools import islice

import torchio as tio
from concurrent.futures import ThreadPoolExecutor
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class MyQueue(tio.Queue):
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
        super(MyQueue, self).__init__(            
            subjects_dataset,
            max_length,
            samples_per_volume,
            sampler,
            num_workers,
            shuffle_subjects,
            shuffle_patches,
            start_background,
            verbose)
        
        self.tpool = ThreadPoolExecutor(max_workers=2)
    
    def fetch_next(self,):
        subject = self._get_next_subject()
        iterable = self.sampler(subject)
        patches = list(islice(iterable, self.samples_per_volume))
        self.patches_list.extend(patches)
        print('fetch_next')
    def __getitem__(self, _):
        if not self.patches_list:
            self.fetch_next()
        sample_patch = self.patches_list.pop()
        print('fetch_next-1')
        if len(self.patches_list) < self.max_length:
            self.tpool.submit(self.fetch_next)
        print('fetch_next-2')
        self.num_sampled_patches += 1
        return sample_patch
    