import warnings
from typing import Tuple

import torch
import numpy as np
import torchio as tio

# import tio.inference.GridSampler as GridSampler 
from scipy.ndimage.filters import gaussian_filter

class GridAggregator(tio.inference.GridAggregator):
    r"""Aggregate patches for dense inference.

    This class is typically used to build a volume made of patches after
    inference of batches extracted by a :class:`~torchio.data.GridSampler`.

    Args:
        sampler: Instance of :class:`~torchio.data.GridSampler` used to
            extract the patches.
        overlap_mode: If ``'crop'``, the overlapping predictions will be
            cropped. If ``'average'``, the predictions in the overlapping areas
            will be averaged with equal weights. See the
            `grid aggregator tests`_ for a raw visualization of both modes.

    .. _grid aggregator tests: https://github.com/fepegar/torchio/blob/master/tests/data/inference/test_aggregator.py

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch-based sampling.
    """  # noqa: E501
    def __init__(self, sampler, overlap_mode: str = 'crop', **kargs):
        subject = sampler.subject
        self.volume_padded = sampler.padding_mode is not None
        self.spatial_shape = subject.spatial_shape
        self._output_tensor = None
        self.patch_overlap = sampler.patch_overlap
        self.parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor = None
        
        if self.overlap_mode == 'gaussian':
            print('init gaussian mask')
            sigma_scale = kargs.get('sigma_scale', 1./8)
            self._gaussian_importance_map = self._get_gaussian(sampler.patch_size, sigma_scale)
        
    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])
        return torch.from_numpy(gaussian_importance_map)
    
    @staticmethod
    def parse_overlap_mode(overlap_mode):
        if overlap_mode not in ('crop', 'average', 'gaussian'):
            message = (
                'Overlap mode must be "crop" or "average" or "gaussian" but '
                f' "{overlap_mode}" was passed'
            )
            raise ValueError(message)

   

    def add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
            ) -> None:
        """Add batch processed by a CNN to the output prediction volume.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. ``batch['image'][torchio.DATA]``.
            locations: 2D tensor with shape :math:`(B, 6)` representing the
                patch indices in the original image. They are typically
                extracted using ``batch[torchio.LOCATION]``.
        """
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        self.initialize_output_tensor(batch)
        if self.overlap_mode == 'crop':
            cropped_patches, crop_locations = self.crop_batch(
                batch,
                locations,
                self.patch_overlap,
            )
            for patch, crop_location in zip(cropped_patches, crop_locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = crop_location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] = patch
        elif self.overlap_mode == 'average':
            self.initialize_avgmask_tensor(batch)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += patch
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += 1
        elif self.overlap_mode == 'gaussian':
            self.initialize_avgmask_tensor(batch)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += patch * self._gaussian_importance_map
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += self._gaussian_importance_map
    def get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        if self._output_tensor.dtype == torch.int64:
            message = (
                'Medical image frameworks such as ITK do not support int64.'
                ' Casting to int32...'
            )
            warnings.warn(message, RuntimeWarning)
            self._output_tensor = self._output_tensor.type(torch.int32)
        if self.overlap_mode == 'average' or self.overlap_mode == 'gaussian':
            # true_divide is used instead of / in case the PyTorch version is
            # old and one the operands is int:
            # https://github.com/fepegar/torchio/issues/526
            output = torch.true_divide(
                self._output_tensor, self._avgmask_tensor)
        else:
            output = self._output_tensor
        if self.volume_padded:
            from ...transforms import Crop
            border = self.patch_overlap // 2
            cropping = border.repeat(2)
            crop = Crop(cropping)
            return crop(output)
        else:
            return output