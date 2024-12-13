import tensorflow as tf
import numpy
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
# Define the feature description for parsing
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),  # Image is stored as bytes
    'BigEarthNet-19_labels': tf.io.VarLenFeature(tf.string),  # Labels are stored as strings
    'BigEarthNet-19_labels_multi_hot': tf.io.FixedLenFeature([19], tf.int64),  # Multi-hot labels (size = number of classes)
    'patch_name': tf.io.VarLenFeature(dtype=tf.string),  # Patch name is stored as string
}

def _parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode image
    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image, [120, 120, 3])  # Reshape according to your image dimensions
    
    # Decode labels
    #labels = parsed_features['BigEarthNet-19_labels'].values
    labels = tf.sparse.to_dense(parsed_features['BigEarthNet-19_labels'])
    # Decode multi-hot labels
    multi_hot_labels = parsed_features['BigEarthNet-19_labels_multi_hot']
    
    # Decode patch name
    patch_name = parsed_features['patch_name'].values
    #patch_name = patch_name.numpy().astype(str)
    return image, labels, multi_hot_labels, patch_name

# Read and parse the TFRecord
def read_tfrecord(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    #raw_record = raw_record in raw_dataset.take(1)
    parsed_dataset = raw_dataset.map(_parse_function)

    for image, labels, multi_hot_labels, patch_name in parsed_dataset.take(1):
        #image = image.numpy()
        # patch_name_str = patch_name.numpy().astype(str)
        # labels_str = labels.numpy().astype(str)
        #image = image[:, :, 0]
        print("Patch Name:", patch_name)
        print("Labels:", labels)
        print("Multi-Hot Labels:", multi_hot_labels.numpy())
        print("Image:", image)

# Example usage
read_tfrecord("/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/splits/train.tfrecord")

# raw_dataset = tf.data.TFRecordDataset("/ds/images/BigEarthNet/BigEarthNet-S2/tf_records_19_labels_rgb/train.tfrecord")

# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)