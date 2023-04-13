from .transformation import Normalize
from .transformation import ZScoreNormalize
from .transformation import ToImage
from .transformation import ToTensor
from .transformation import RandomHorizontalFlip
from .transformation import RandomVerticalFlip
from .transformation import RandomRotate
from .transformation import RandomScale
from .transformation import RandomColorJitter
from .transformation import RandomSliceSelect

import numpy as np
import os
import re
from glob import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import LightningDataModule

class CKBrainMetDataset(Dataset):

    def __init__(self, config, mode, patient_paths, transform, image_size):
        super().__init__()
        assert mode in ['train', 'test']
        """
        if mode == train       -> output only normal images without label
        if mode == test        -> output both normal and abnormal images with label
        """
        self.config = config
        self.mode = mode
        self.patient_paths = patient_paths
        self.transform = transform
        self.image_size = image_size
        self.files = self.build_file_paths(self.patient_paths)

    def build_file_paths(self, patient_paths):

        files = []

        for patient_path in patient_paths:
            file_paths = glob(os.path.join(patient_path + "/*" + self.config.dataset.select_slice + ".npy")) #指定のスライスのパスを取得
            for file_path in file_paths:
                
                if 'Abnormal' in file_path:
                    class_name = 'Abnormal'
                else:
                    #assert 'normal' in file_name
                    class_name = 'Normal'

                patient_id = patient_path.split('/')[-1]
                file_name = file_path.split('/')[-1]
                study_name = self.get_study_name(patient_path)
                slice_num = self.get_slice_num(file_name)

                if self.mode == 'train':
                    files.append({
                        'image': file_path,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                    })

                elif self.mode == 'test' or self.mode == 'test_normal':
                    label_path = self.get_label_path(file_path)

                    files.append({
                        'image': file_path,
                        'label': label_path,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                    })

        return files

    def get_study_name(self, patient_path):
        study_name = patient_path.split('/')[-3]
        return study_name
    
    def get_slice_num(self, file_name):
        n = re.findall(r'\d+', file_name) #image_fileのスライス番号の取り出し
        return n[-1]

    def get_label_path(self, file_path):
        file_path = file_path.replace(self.config.dataset.select_slice, 'seg')
        return file_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = np.load(self.files[index]['image'])
        image = np.flipud(np.transpose(image))

        sample = {
            'image': image.astype(np.float32),
            'patient_id': self.files[index]['patient_id'],
            'class_name': self.files[index]['class_name'],
            'study_name': self.files[index]['study_name'],
            'slice_num': self.files[index]['slice_num'],
        }

        if self.mode == 'test':
            if os.path.exists(self.files[index]['label']):
                label = np.load(self.files[index]['label'])
                label = np.flipud(np.transpose(label))
            else:
                label = np.zeros_like(image)

            sample.update({
                'label': label.astype(np.int32),
            })

        if self.transform:
            sample = self.transform(sample)

        return sample


class CKBrainMetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root_dir_path = self.config.dataset.root_dir_path
        self.CKBrainMetDataset = CKBrainMetDataset
        self.omit_transform = False

    def get_patient_paths(self, base_dir_path):
        patient_ids = os.listdir(base_dir_path)
        return [os.path.join(base_dir_path, p) for p in patient_ids]

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            
            if self.config.dataset.use_augmentation:
                transform = transforms.Compose([
                    ToImage(),
                    RandomHorizontalFlip(),
                    RandomRotate(degree=20),
                    RandomScale(mean=1.0, var=0.05, image_fill=0),
                    # RandomColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                ])

            val_transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                ])

            if self.omit_transform:
                transform = None
            
            train_patient_paths = self.get_patient_paths(os.path.join(self.root_dir_path, 'MICCAI_BraTS_2019_Data_Val_Testing/Normal'))
            self.train_dataset = self.CKBrainMetDataset(config=self.config, mode='train', patient_paths=train_patient_paths, transform=transform, image_size=self.config.dataset.image_size)
            self.valid_dataset = self.CKBrainMetDataset(config=self.config, mode='train', patient_paths=train_patient_paths, transform=val_transform, image_size=self.config.dataset.image_size)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                    Normalize(min_val=0, max_val=255),
                ])
            test_patient_paths = self.get_patient_paths(os.path.join(self.root_dir_path, 'MICCAI_BraTS_2019_Data_Training/Abnormal'))
            self.test_dataset = self.CKBrainMetDataset(config=self.config, mode='test', patient_paths=test_patient_paths, transform=transform, image_size=self.config.dataset.image_size)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.dataset.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.dataset.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.dataset.batch_size, shuffle=False)