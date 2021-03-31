# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:58:21 2021

@author: remco
"""
import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import matplotlib.pyplot as plt

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
        CamVidClass('Void', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CamVidClass('Animal', 1, 255, 'void', 0, True, True, (64, 128, 64)),
        CamVidClass('Archway', 2, 255, 'void', 0, True, True, (192, 0, 128)),
        CamVidClass('Bicyclist', 3, 0, 'Bicyclist', 1, True, False, (0, 128, 192)),  #
        CamVidClass('Bridge', 4, 255, 'void', 0, True, True, (0, 128, 64)),
        CamVidClass('Building', 5, 1, 'Building', 2, True, False, (128,0,0)), #
        CamVidClass('Car', 6, 2, 'Car', 3, True, False, (64,0,128)), #
        CamVidClass('CartLuggagePram', 7, 255, 'void', 0, False, True, (64,0,192)),
        CamVidClass('Child', 8, 255, 'void', 0, False, True, (192,128,64)),
        CamVidClass('Column_pole', 9, 3, 'Pole', 4, True, False, (192,192,128)), #
        CamVidClass('Fence', 10, 4, 'Fence', 5, True, False, (64,64,128)), #
        CamVidClass('LaneMkgsDriv', 11, 255, 'void', 0, True, True, (128,0,192)),
        CamVidClass('LaneMkgsNonDriv', 12, 255, 'void', 0, True, True, (192,0,64)),
        CamVidClass('Misc_Text', 13, 255, 'void', 0, False, True, (128,128,64)),
        CamVidClass('MotorcycleScooter', 14, 255, 'void', 0, True, True, (192,0,192)),
        CamVidClass('OtherMoving', 15, 255, 'void', 0, True, True, (128,64,64)),
        CamVidClass('ParkingBlock', 16, 255, 'void', 0, True, True, (64,192,128)),
        CamVidClass('Pedestrian', 17, 5, 'Pedestrian', 6, True, False,  (64,64,0)), #
        CamVidClass('Road', 18, 6, 'Road', 7, True, False, (128,64,128)), #
        CamVidClass('RoadShoulder', 19, 255, 'void', 0, False, True, (128,128,192)),
        CamVidClass('Sidewalk', 20, 7, 'Sidewalk', 8, True, False, (0,0,192)), #
        CamVidClass('SignSymbol', 21, 8, 'SignSymbol', 9, True, False, (192,128,128)), #
        CamVidClass('Sky', 22, 9, 'Sky', 10, True, False, (128,128,128)), #
        CamVidClass('SUVPickupTruck', 23, 255, 'void', 0, False, True, (64,128,192)),
        CamVidClass('TrafficCone', 24, 255, 'void', 0, True, True, (0,0,64)),
        CamVidClass('TrafficLight', 25, 255, 'void', 0, True, True, (0,64,64)),
        CamVidClass('Train', 26, 255, 'void', 0, True, True, (192,64,128)),
        CamVidClass('Tree', 27, 10, 'Tree', 11, True, False, (128,128,0)), #
        CamVidClass('Truck_Bus', 28, 255, 'void', 0, True, True, (192,128,192)),
        CamVidClass('Tunnel', 29, 255, 'void', 0, True, True, (64,0,64)),
        CamVidClass('VegetationMisc', 30, 255, 'void', 0, True, True, (192,192,0)),
        CamVidClass('Wall', 31, 255, 'void', 0, True, True, (64,192,0)),
        
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

    
        for file_name in os.listdir(self.images_dir):
                #target_types = []
                #for t in self.target_type:
                target_name = (os.path.splitext(file_name)[0])
                new_name ="_".join([target_name,"L.png"])
                target_types = (os.path.join(self.targets_dir, new_name))
                    
                #print(target_types)

                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(target_types)
                
        #print(self.targets[1])

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
        target = Image.open(self.targets[index])

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


    def _get_target_suffix(self, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'
        elif target_type == 'semantic':
            return '{}_labelIds.png'
        elif target_type == 'color':
            return '{}_color.png'
        else:
            return '{}_polygons.json'
    

