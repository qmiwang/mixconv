import matplotlib.pyplot as plt
import numpy as np
from torchio.transforms.preprocessing.intensity import NormalizationTransform
import torch

class ZNormalization(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            mean_std = None,
            masking_method = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ('masking_method',)
        self.mean_std = mean_std

    def apply_normalization(
            self,
            subject,
            image_name: str,
            mask,
            ) -> None:
        image = subject[image_name]
        standardized = self.znorm(
            image.data,
            mask,
            self.mean_std,
        )
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        image.set_data(standardized)

    @staticmethod
    def znorm(tensor, mask, mean_std = None) -> torch.Tensor:
        tensor = tensor.clone().float()
        if mean_std is None:
            values = tensor.masked_select(mask)
            mean, std = values.mean(), values.std()
        else:
            mean, std = torch.Tensor(mean_std).float()
        if std == 0:
            return None
        tensor -= mean
        tensor /= std
        return tensor
