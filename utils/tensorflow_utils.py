# import skimage.io
# import tensorflow as tf
# import numpy as np
# import os
# import json
# from PIL import Image

# def prep_example(image_data, BigEarthNet_19_labels, BigEarthNet_19_labels_multi_hot, patch_name):
#     return tf.train.Example(
#         features=tf.train.Features(
#             feature={
#                 'image': tf.train.Feature(
#                     bytes_list=tf.train.BytesList(value=[image_data.tobytes()])),
#                 'BigEarthNet-19_labels': tf.train.Feature(
#                     bytes_list=tf.train.BytesList(
#                         value=[i.encode('utf-8') for i in BigEarthNet_19_labels])),
#                 'BigEarthNet-19_labels_multi_hot': tf.train.Feature(
#                     int64_list=tf.train.Int64List(value=BigEarthNet_19_labels_multi_hot)),
#                 'patch_name': tf.train.Feature(
#                     bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
#             }))

# def create_split(root_folder, label_folder, patch_names, TFRecord_writer, label_indices, UPDATE_JSON):
#     label_conversion = label_indices['label_conversion']
#     BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].items()}
    
#     progress_bar = tf.keras.utils.Progbar(target=len(patch_names))
#     for patch_idx, patch_name in enumerate(patch_names):
#         patch_folder_path = os.path.join(root_folder, patch_name)
#         image_path = os.path.join(patch_folder_path + '.jpg')
        
#         # Load the JPEG image
#         image = skimage.io.imread(image_path)
#         image_data = np.array(image)
#         print(image_data.shape)
#         original_labels_multi_hot = np.zeros(
#             len(label_indices['original_labels'].keys()), dtype=int)
#         BigEarthNet_19_labels_multi_hot = np.zeros(len(label_conversion), dtype=int)
#         patch_json_root = os.path.join(label_folder, patch_name)
#         patch_json_path = os.path.join(patch_json_root, patch_name + '_labels_metadata.json')

#         with open(patch_json_path, 'rb') as f:
#             patch_json = json.load(f)

#         original_labels = patch_json['labels']
#         for label in original_labels:
#             original_labels_multi_hot[label_indices['original_labels'][label]] = 1

#         for i in range(len(label_conversion)):
#             BigEarthNet_19_labels_multi_hot[i] = (
#                 np.sum(original_labels_multi_hot[label_conversion[i]]) > 0
#             ).astype(int)

#         BigEarthNet_19_labels = []
#         for i in np.where(BigEarthNet_19_labels_multi_hot == 1)[0]:
#             BigEarthNet_19_labels.append(BigEarthNet_19_label_idx[i])

#         if UPDATE_JSON:
#             patch_json['BigEarthNet_19_labels'] = BigEarthNet_19_labels
#             with open(patch_json_path, 'w') as f:  # Changed 'wb' to 'w' for writing JSON files
#                 json.dump(patch_json, f)

#         example = prep_example(
#             image_data, 
#             BigEarthNet_19_labels,
#             BigEarthNet_19_labels_multi_hot,
#             patch_name
#         )
#         TFRecord_writer.write(example.SerializeToString())
#         progress_bar.update(patch_idx)

# def prep_tf_record_files(root_folder, out_folder, label_folder, split_names, patch_names_list, label_indices, UPDATE_JSON):
#     try:
#         writer_list = []
#         for split_name in split_names:
#             writer_list.append(
#                 tf.io.TFRecordWriter(os.path.join(
#                     out_folder, split_name + '.tfrecord'))
#             )
#     except:
#         print('ERROR: TFRecord writer is not able to write files')
#         exit()

#     for split_idx in range(len(patch_names_list)):
#         print('INFO: creating the split of', split_names[split_idx], 'is started')
#         create_split(
#             root_folder,
#             label_folder,
#             patch_names_list[split_idx], 
#             writer_list[split_idx],
#             label_indices,
#             UPDATE_JSON
#         )
#         writer_list[split_idx].close()
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def prep_example(image_data, BigEarthNet_19_labels, BigEarthNet_19_labels_multi_hot, patch_name):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_data.tobytes()])),
                'BigEarthNet-19_labels': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[i.encode('utf-8') for i in BigEarthNet_19_labels])),
                'BigEarthNet-19_labels_multi_hot': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=BigEarthNet_19_labels_multi_hot)),
                'patch_name': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
            }))

def process_batch(patch_names, root_folder, label_folder, label_indices, UPDATE_JSON):
    BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].items()}
    examples = []
    
    for patch_name in patch_names:
        patch_folder_path = os.path.join(root_folder, patch_name)
        image_path = os.path.join(patch_folder_path + '.jpg')
        
        # Load and process the JPEG image
        image = Image.open(image_path)
        image_data = np.array(image)

        original_labels_multi_hot = np.zeros(len(label_indices['original_labels'].keys()), dtype=int)
        BigEarthNet_19_labels_multi_hot = np.zeros(len(label_indices['label_conversion']), dtype=int)
        
        patch_json_root = os.path.join(label_folder, patch_name)
        patch_json_path = os.path.join(patch_json_root, patch_name + '_labels_metadata.json')

        with open(patch_json_path, 'r') as f:
            patch_json = json.load(f)

        original_labels = patch_json['labels']
        for label in original_labels:
            original_labels_multi_hot[label_indices['original_labels'][label]] = 1

        for i in range(len(label_indices['label_conversion'])):
            BigEarthNet_19_labels_multi_hot[i] = (
                np.sum(original_labels_multi_hot[label_indices['label_conversion'][i]]) > 0
            ).astype(int)

        BigEarthNet_19_labels = []
        for i in np.where(BigEarthNet_19_labels_multi_hot == 1)[0]:
            BigEarthNet_19_labels.append(BigEarthNet_19_label_idx[i])

        if UPDATE_JSON:
            patch_json['BigEarthNet_19_labels'] = BigEarthNet_19_labels
            with open(patch_json_path, 'w') as f:
                json.dump(patch_json, f)

        example = prep_example(
            image_data, 
            BigEarthNet_19_labels,
            BigEarthNet_19_labels_multi_hot,
            patch_name
        )
        examples.append(example.SerializeToString())

    return examples

def create_split(root_folder, label_folder, patch_names, TFRecord_writer, label_indices, UPDATE_JSON, batch_size=32):
    progress_bar = tf.keras.utils.Progbar(target=len(patch_names))
    
    for start_idx in range(0, len(patch_names), batch_size):
        batch_patch_names = patch_names[start_idx:start_idx + batch_size]
        
        # Process the batch of patches
        examples = process_batch(batch_patch_names, root_folder, label_folder, label_indices, UPDATE_JSON)
        
        # Write all examples in the batch to TFRecord
        for example in examples:
            TFRecord_writer.write(example)

        progress_bar.update(start_idx + batch_size)

def prep_tf_record_files(root_folder, out_folder, label_folder, split_names, patch_names_list, label_indices, UPDATE_JSON, batch_size=32):
    try:
        writer_list = [tf.io.TFRecordWriter(os.path.join(out_folder, split_name + '.tfrecord')) for split_name in split_names]
    except Exception as e:
        print('ERROR: TFRecord writer is not able to write files:', str(e))
        exit()

    for split_idx in range(len(patch_names_list)):
        print('INFO: creating the split of', split_names[split_idx], 'is started')
        create_split(
            root_folder,
            label_folder,
            patch_names_list[split_idx], 
            writer_list[split_idx],
            label_indices,
            UPDATE_JSON,
            batch_size
        )
        writer_list[split_idx].close()
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def prep_example(image_data, BigEarthNet_19_labels, BigEarthNet_19_labels_multi_hot, patch_name):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_data.tobytes()])),
                'BigEarthNet-19_labels': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[i.encode('utf-8') for i in BigEarthNet_19_labels])),
                'BigEarthNet-19_labels_multi_hot': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=BigEarthNet_19_labels_multi_hot)),
                'patch_name': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
            }))

def process_patch(patch_name, root_folder, label_folder, label_indices, UPDATE_JSON):
    BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].items()}
    
    patch_folder_path = os.path.join(root_folder, patch_name)
    image_path = os.path.join(patch_folder_path + '.jpg')
    
    # Load and process the JPEG image
    image = Image.open(image_path)
    image_data = np.array(image)

    original_labels_multi_hot = np.zeros(len(label_indices['original_labels'].keys()), dtype=int)
    BigEarthNet_19_labels_multi_hot = np.zeros(len(label_indices['label_conversion']), dtype=int)
    
    patch_json_root = os.path.join(label_folder, patch_name)
    patch_json_path = os.path.join(patch_json_root, patch_name + '_labels_metadata.json')

    with open(patch_json_path, 'r') as f:
        patch_json = json.load(f)

    original_labels = patch_json['labels']
    for label in original_labels:
        original_labels_multi_hot[label_indices['original_labels'][label]] = 1

    for i in range(len(label_indices['label_conversion'])):
        BigEarthNet_19_labels_multi_hot[i] = (
            np.sum(original_labels_multi_hot[label_indices['label_conversion'][i]]) > 0
        ).astype(int)

    BigEarthNet_19_labels = [BigEarthNet_19_label_idx[i] for i in np.where(BigEarthNet_19_labels_multi_hot == 1)[0]]

    if UPDATE_JSON:
        patch_json['BigEarthNet_19_labels'] = BigEarthNet_19_labels
        with open(patch_json_path, 'w') as f:
            json.dump(patch_json, f)

    example = prep_example(
        image_data, 
        BigEarthNet_19_labels,
        BigEarthNet_19_labels_multi_hot,
        patch_name
    )
    return example.SerializeToString()

def process_batch(patch_names, root_folder, label_folder, label_indices, UPDATE_JSON):
    examples = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_patch, patch_name, root_folder, label_folder, label_indices, UPDATE_JSON): patch_name for patch_name in patch_names}
        for future in as_completed(futures):
            examples.append(future.result())
    return examples

def create_split(root_folder, label_folder, patch_names, TFRecord_writer, label_indices, UPDATE_JSON, batch_size=32):
    progress_bar = tf.keras.utils.Progbar(target=len(patch_names))
    
    for start_idx in range(0, len(patch_names), batch_size):
        batch_patch_names = patch_names[start_idx:start_idx + batch_size]
        
        # Process the batch of patches
        examples = process_batch(batch_patch_names, root_folder, label_folder, label_indices, UPDATE_JSON)
        
        # Write all examples in the batch to TFRecord
        for example in examples:
            TFRecord_writer.write(example)

        progress_bar.update(start_idx + batch_size)

def prep_tf_record_files(root_folder, out_folder, label_folder, split_names, patch_names_list, label_indices, UPDATE_JSON, batch_size=32):
    try:
        writer_list = [tf.io.TFRecordWriter(os.path.join(out_folder, split_name + '.tfrecord')) for split_name in split_names]
    except Exception as e:
        print('ERROR: TFRecord writer is not able to write files:', str(e))
        exit()

    for split_idx in range(len(patch_names_list)):
        print(f'INFO: Creating the split of {split_names[split_idx]} has started')
        create_split(
            root_folder,
            label_folder,
            patch_names_list[split_idx], 
            writer_list[split_idx],
            label_indices,
            UPDATE_JSON,
            batch_size
        )
        writer_list[split_idx].close()
        print(f'INFO: Finished creating the split of {split_names[split_idx]}')
