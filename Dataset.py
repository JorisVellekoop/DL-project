# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:58:21 2021

@author: remco
"""
import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CamVid(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Examples:
        Get semantic segmentation target
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')
            img, smnt = dataset[0]
        Get multiple targets
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])
            img, (inst, col, poly) = dataset[0]
        Validate on the "coarse" set
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')
            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CamVidClass = namedtuple('Class', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CamVidClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CamVidClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CamVidClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CamVidClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CamVidClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CamVidClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CamVidClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CamVidClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CamVidClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CamVidClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CamVidClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CamVidClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CamVidClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CamVidClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CamVidClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CamVidClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CamVidClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CamVidClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CamVidClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CamVidClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CamVidClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CamVidClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CamVidClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CamVidClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CamVidClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CamVidClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CamVidClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CamVidClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CamVidClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CamVidClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CamVidClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CamVidClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CamVidClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CamVidClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CamVidClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(CamVid, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, split, 'images')
        self.targets_dir = os.path.join(self.root, split, 'labels')
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        
        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        
        for file_name in os.listdir(self.images_dir):
                target_types = []
                for t in self.target_type:
                    target_name = (file_name.split('_leftImg8bit')[0])
                    target_types.append(os.path.join(self.targets_dir, target_name))

                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(target_types)
                print(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'
        elif target_type == 'semantic':
            return '{}_labelIds.png'
        elif target_type == 'color':
            return '{}_color.png'
        else:
            return '{}_polygons.json'
        
        
camvid = CamVid('CamVidData_2','train')

a = camvid.__getitem__(1)