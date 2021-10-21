import numpy as np
import os
from tqdm import tqdm
import logging
import torchio as tio
import utils
import random


class Loader3D():

    def __init__(self, config, do_train=True, additional_dataset=False, is_competitor=False, skip_primary=False):

        self.config = config

        self.subjects = {
            'train': [],
            'syntetic': [],
            'test': [],
            'val': []
        }

        splits = {}
        if do_train:
            if not skip_primary:
                splits['train'] = 'sparse' if is_competitor else 'dense'
            if additional_dataset:
                splits['syntetic'] = 'sparse'
        splits['val'] = 'sparse' if is_competitor else 'dense'
        splits['test'] = 'dense'  # always!
        for split, dataset_type in splits.items():
            logging.info(f"loading {split} dataset from {os.path.join(config['file_path'], dataset_type, split)}")
            tmp_gt = np.load(os.path.join(config['file_path'], dataset_type, split, 'gt.npy'), allow_pickle=True)
            tmp_data = np.load(os.path.join(config['file_path'], dataset_type, split, 'data.npy'), allow_pickle=True)
            for p in tqdm(range(tmp_gt.shape[0])):
                assert np.max(tmp_data[p]) <= 1  # data should be normalized by default
                assert np.unique(tmp_gt[p]).size <= len(self.config['labels'])
                label = tio.LabelMap(tensor=tmp_gt[p].astype(np.uint8)[None]) if split in ['train', 'syntetic'] else tmp_gt[p].astype(np.uint8)
                self.subjects[split].append(tio.Subject(
                    data=tio.ScalarImage(tensor=tmp_data[p][None].astype(np.float)),
                    label=label,
                ))

        # if do_train:
        #     # PRE-TRAINING
        #     if additional_dataset:
        #         logging.info(f"using additional dataset")
        #         tmp_gt = np.load(os.path.join(config['file_path'], 'sparse', 'syntetic', 'gt.npy'))
        #         tmp_data = np.load(os.path.join(config['file_path'], 'sparse', 'syntetic', 'data.npy'))
        #         for p in tqdm(range(tmp_gt.shape[0])):
        #             self.subjects['syntetic'].append(tio.Subject(
        #                 data=tio.ScalarImage(tensor=tmp_data[p][None].astype(np.float)),
        #                 label=tio.LabelMap(tensor=tmp_gt[p].astype(np.uint8)[None]),
        #         ))
        #     else:
        #         logging.info("additional dataset SKIPPED here")
        #
        #     # TRAINING & VAL
        #     for split in ['train', 'val', 'test']:
        #         subdir = 'sparse' if is_competitor and split != 'test' else 'dense'
        #         logging.info(f"loading {split} dataset from {os.path.join(config['file_path'], subdir, split)}")
        #         tmp_gt = np.load(os.path.join(config['file_path'], subdir, split, 'gt.npy'),  allow_pickle=True)
        #         tmp_data = np.load(os.path.join(config['file_path'], subdir, split, 'data.npy'), allow_pickle=True)
        #         for p in tqdm(range(tmp_gt.shape[0])):
        #             assert np.max(tmp_data[p]) <= 1  # data should be normalized by default
        #             assert np.unique(tmp_gt[p]).size <= len(self.config['labels'])
        #             label = tio.LabelMap(tensor=tmp_gt[p].astype(np.uint8)[None]) if split == 'train' else tmp_gt[p].astype(np.uint8)
        #             self.subjects[split].append(tio.Subject(
        #                 data=tio.ScalarImage(tensor=tmp_data[p][None].astype(np.float)),
        #                 label=label
        #             ))

        self.do_train = do_train
        self.additional_dataset = additional_dataset

        aug_filepath = config.get("augmentations_file", None)
        auglist = [] if aug_filepath is None else utils.load_config_yaml(aug_filepath)
        augment = AugFactory(auglist)
        augment.log()  # write what we are using to logfile
        self.transforms = augment.get_transform()

    def get_aggregator(self):
        sampler = self.get_sampler()
        return tio.inference.GridAggregator(sampler)

    def get_sampler(self):
        return tio.GridSampler(patch_size=(32, 32, 32), patch_overlap=0)

    def split_dataset(self, rank=0, world_size=1):
        training_set = self.subjects['train'] + self.subjects['syntetic']
        random.shuffle(training_set)
        train = tio.SubjectsDataset(training_set[rank::world_size], transform=self.transforms) if self.do_train else None

        test = [tio.GridSampler(subject, patch_size=(32, 32, 32), patch_overlap=0) for subject in self.subjects['test']]
        val = [tio.GridSampler(subject, patch_size=(32, 32, 32), patch_overlap=0) for subject in self.subjects['val']]
        return train, test, val


class AugFactory:
    def __init__(self, aug_list):
        self.aug_list = aug_list
        self.transforms = self.factory(self.aug_list, [])

    def log(self):
        """
        save the list of aug for this experiment to the default log file
        :param path:
        :return:
        """
        logging.info('going to use the following augmentations:: %s', self.aug_list)

    def factory(self, auglist, transforms):
        for aug in auglist:
            if aug == 'OneOf':
                transforms.append(tio.OneOf(self.factory(auglist[aug], [])))
            else:
                try:
                    kwargs = {}
                    for param, value in auglist[aug].items():
                        kwargs[param] = value
                    transforms.append(getattr(tio, aug)(**kwargs))
                except:
                    raise Exception(f"this transform is not valid: {aug}")
        return transforms

    def get_transform(self):
        """
        return the transform object
        :return:
        """
        return tio.Compose(self.transforms)