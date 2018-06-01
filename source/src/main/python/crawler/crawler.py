_author__ = 'MSteger'

import os
import glob
import numpy as np
from shutil import copyfile
from itertools import compress
from torchvision import transforms
from PIL import Image
from joblib import Parallel, delayed
from google_images_download import google_images_download

def run_crawler(arguments):
    loader = google_images_download.googleimagesdownload()
    return loader.download(arguments)

def process_img(img_path, transformer, output_dir, output_filename = None):
    # add options for flips, crops & color jitter!
    try:
        img = Image.open(img_path).convert('RGB')
        new_img = transformer(img)
        new_img_filename = os.path.split(img_path)[-1]
        if output_filename is not None:
            ext = os.path.splitext(os.path.split(img_path)[-1])[-1]
            new_img_filename = '{}_{}{}'.format(output_dir.split('/')[-2], output_filename,ext.lower())
        new_img_path = os.path.join(output_dir, new_img_filename)
        new_img.save(new_img_path)
        print 'saved {} to {}!'.format(img_path, new_img_path)
    except Exception as e:
        print 'failure: {} - {}'.format(img_path, e)
    return

def run_preprocessing(img_paths, transformer, output_dir, n_jobs = 1, index_imgs = False):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    return Parallel(n_jobs = n_jobs, backend="multiprocessing")(delayed(process_img)(img_path = img_path, transformer = transformer, output_dir = output_dir, output_filename = idx if index_imgs else None) for idx, img_path in enumerate(img_paths))

def sample_from_tinyImageNet(data_dir, size, exclude_class = None, include_class = None, class_prob = None):
    train_samples = []
    if include_class is None: include_class = [dir for dir in os.listdir(data_dir)] if class_prob is None else class_prob.keys()
    if class_prob is None: class_prob = {dir: 1./len(include_class) for dir in include_class} # equally distributed by default

    if exclude_class is not None:
        for ex_class in exclude_class: class_prob.pop(ex_class, None)

    for sample_class, sample_prob in class_prob.items():
        try:
            sample_class_path = os.path.join(data_dir, sample_class)
            class_size = int(sample_prob * size)
            all_class_samples = [f for f in glob.glob('{}/*.*'.format(sample_class_path)) if f.lower().endswith(('.jpeg', '.jpg', '.jpeg', '.png'))]
            random_selection = np.random.randint(0, len(all_class_samples), class_size)
            train_samples += list(compress(all_class_samples, random_selection))
        except Exception as e:
            print 'failure! {}'.format(e)
    return train_samples

def cp_file(old_path, new_path):
    print 'cp {} to {}'.format(old_path, new_path)
    return copyfile(old_path, new_path)

def copy_files(file_paths, new_directory_path, n_jobs = 1):
    if not os.path.exists(new_directory_path): os.makedirs(new_directory_path)
    return Parallel(n_jobs = n_jobs, backend = 'multiprocessing')(delayed(cp_file)(old_path = old_path, new_path = os.path.join(new_directory_path, '0_{}{}'.format(idx, os.path.splitext(old_path)[-1]))) for idx, old_path in enumerate(file_paths))


if __name__ == '__main__':
    crawler_args = {
        'keywords': 'elefant',
        'limit': 5000,
        'size': 'large',
        'print_urls': False,
        'metadata': False,
        'output_directory': '/media/msteger/storage/resources/DreamPhant/downloads/',
        'no_directory': False,
        'chromedriver': '/home/mks/Downloads/chromedriver',
        'related_images': True,
        'extract_metadata': True
    }
    # run_crawler(arguments = crawler_args)
    raw_crawler_images = r'/media/msteger/storage/resources/DreamPhant/downloads/'
    img_paths = [os.path.join(root, file) for root, dirnames, filenames in os.walk(raw_crawler_images) for file in filenames]

    transformer = transforms.Compose([transforms.Resize((224, 224))])
    # run_preprocessing(img_paths = img_paths, transformer = transformer, output_dir = r'/media/msteger/storage/resources/DreamPhant/1/', n_jobs = -1, index_imgs = True)

    non_Phant_data_path = r'/media/msteger/storage/resources/tiny-imagenet-200/train'
    non_Phant_data = sample_from_tinyImageNet(data_dir = non_Phant_data_path, size = 6000, exclude_class = ['n01522450'], include_class = None, class_prob = None)#{dir: 1./199 for dir in non_Phant_classes})
    copy_files(file_paths = non_Phant_data, new_directory_path = r'/media/msteger/storage/resources/DreamPhant/data/train/0', n_jobs = -1)
    print 'done'