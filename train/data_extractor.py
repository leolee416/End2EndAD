import os
import csv
import random
import cv2
import numpy as np

def parse_driving_data(driving_data_file):
    """Parse the driving data CSV file to extract useful information."""
    driving_data = {}
    with open(driving_data_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key, value = row
            driving_data[key] = value
    # Extract important fields, for example, steering angle
    steering_angle = float(driving_data.get('Steering', 0))
    return steering_angle

def parse_labels(label_file):
    """Parse the label CSV to extract object information (e.g., bounding boxes)."""
    objects = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]  # e.g., 'car', 'truck'
            x1, y1, x2, y2 = map(int, row[1:])  # Bounding box coordinates
            objects.append((label, x1, y1, x2, y2))
    return objects

def preprocess_image(image_path):
    """Preprocess the image (resize and normalize)."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (200, 66))  # Resize to match CNN input
    image = image / 255.0  # Normalize the pixel values
    return image


def batch_generator(images_dir, drivings_dir, labels_dir, batch_size, is_training=True):
    """Generate a batch of image, steering angle, and labels for training."""
    image_paths = sorted(os.listdir(images_dir))
    driving_data_files = sorted(os.listdir(drivings_dir))
    label_files = sorted(os.listdir(labels_dir))

    # Group images and labels by timestamp (first 19 characters of the filename)
    images_by_timestamp = {}
    labels_by_timestamp = {}

    for image_path in image_paths:
        timestamp = image_path[:19]  # Extract first 19 characters as timestamp
        if "M" in image_path:  # Only consider images with 'M' in their filename
            images_by_timestamp[timestamp] = image_path

    for label_path in label_files:
        timestamp = label_path[:19]  # Extract first 19 characters as timestamp
        labels_by_timestamp[timestamp] = label_path

    batch_images = []
    batch_steering_angles = []
    batch_labels = []

    # Keep track of how many samples we have processed
    processed_samples = 0
    total_samples = len(images_by_timestamp)
    print(total_samples)
    ccc = 0

    while ccc < len(driving_data_files):  # Stop after one full pass over the data
        for driving_data_file in driving_data_files:
            ccc += 1
            timestamp = driving_data_file[:19]  # Extract timestamp from driving data file
            
            # Check if there is an 'M' image for this timestamp
            if timestamp not in images_by_timestamp:
                continue

            # Get corresponding image and label for this timestamp
            image_path = images_by_timestamp[timestamp]
            label_path = labels_by_timestamp.get(timestamp)

            # If no matching label exists, skip this timestamp
            if label_path is None:
                print(f"Skipping timestamp {timestamp} due to missing label file")
                continue

            # Process the driving data (steering angle)
            driving_data_path = os.path.join(drivings_dir, driving_data_file)
            steering_angle = parse_driving_data(driving_data_path)

            # Process the image
            image = preprocess_image(os.path.join(images_dir, image_path))

            # Process the labels (bounding boxes)
            objects = parse_labels(os.path.join(labels_dir, label_path))

            # Augmentation (optional, for training only)
            if is_training:
                image, steering_angle = augment(image, steering_angle)

            # Append the image, steering angle, and labels to the batch
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)
            batch_labels.append(objects)

            processed_samples += 1

            # Yield a batch
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_steering_angles), batch_labels
                batch_images = []
                batch_steering_angles = []
                batch_labels = []

    # Handle any remaining data
    if batch_images:
        yield np.array(batch_images), np.array(batch_steering_angles), batch_labels
"""
def batch_generator(images_dir, drivings_dir, labels_dir, batch_size, is_training=True):
    ##Generate a batch of image, steering angle, and labels for training.
    image_paths = sorted(os.listdir(images_dir))
    driving_data_files = sorted(os.listdir(drivings_dir))
    label_files = sorted(os.listdir(labels_dir))

    batch_images = []
    batch_steering_angles = []
    batch_labels = []

    while True:
        for i in range(len(image_paths)):
            image_path = os.path.join(images_dir, image_paths[i])
            driving_data_path = os.path.join(drivings_dir, driving_data_files[i])
            label_path = os.path.join(labels_dir, label_files[i])

            # Process the driving data (extract steering angle)
            steering_angle = parse_driving_data(driving_data_path)

            # Process the image
            image = preprocess_image(image_path)

            # Process the labels (bounding boxes)
            objects = parse_labels(label_path)

            # Augmentation (optional, for training only)
            if is_training:
                image, steering_angle = augment(image, steering_angle)

            batch_images.append(image)
            batch_steering_angles.append(steering_angle)
            batch_labels.append(objects)

            # Yield a batch
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_steering_angles), batch_labels
                batch_images = []
                batch_steering_angles = []
                batch_labels = []
"""
# def batch_generator(images_dir, drivings_dir, labels_dir, batch_size, is_training=True):
#     """Generate a batch of image, steering angle, and labels for training."""
#     image_paths = sorted(os.listdir(images_dir))
#     driving_data_files = sorted(os.listdir(drivings_dir))
#     label_files = sorted(os.listdir(labels_dir))

#     # Group images and labels by timestamp (first 19 characters of the filename)
#     images_by_timestamp = {}
#     labels_by_timestamp = {}

#     for image_path in image_paths:
#         timestamp = image_path[:19]  # Extract first 19 characters as timestamp
#         if timestamp not in images_by_timestamp:
#             images_by_timestamp[timestamp] = []
#         images_by_timestamp[timestamp].append(image_path)

#     for label_path in label_files:
#         timestamp = label_path[:19]  # Extract first 19 characters as timestamp
#         if timestamp not in labels_by_timestamp:
#             labels_by_timestamp[timestamp] = []
#         labels_by_timestamp[timestamp].append(label_path)

#     batch_images = []
#     batch_steering_angles = []
#     batch_labels = []

#     # Keep track of how many samples we have processed
#     processed_samples = 0
#     total_samples = sum(len(images_by_timestamp[timestamp]) for timestamp in images_by_timestamp)

#     while processed_samples < total_samples:  # Stop after one full pass over the data
#         for driving_data_file in driving_data_files:
#             timestamp = driving_data_file[:19]  # Extract timestamp from driving data file
            
#             # Get corresponding images and labels for this timestamp
#             image_paths_for_timestamp = images_by_timestamp.get(timestamp, [])
#             label_paths_for_timestamp = labels_by_timestamp.get(timestamp, [])

#             # If the number of images and labels for the timestamp is not equal, skip this timestamp
#             if len(image_paths_for_timestamp) != len(label_paths_for_timestamp):
#                 print(f"Skipping timestamp {timestamp} due to mismatch between images and labels")
#                 continue

#             # Process the driving data (steering angle)
#             driving_data_path = os.path.join(drivings_dir, driving_data_file)
#             steering_angle = parse_driving_data(driving_data_path)

#             # Process all images and labels for the current timestamp
#             for i in range(len(image_paths_for_timestamp)):
#                 image_path = os.path.join(images_dir, image_paths_for_timestamp[i])
#                 label_path = os.path.join(labels_dir, label_paths_for_timestamp[i])

#                 # Process the image
#                 image = preprocess_image(image_path)

#                 # Process the labels (bounding boxes)
#                 objects = parse_labels(label_path)

#                 # Augmentation (optional, for training only)
#                 if is_training:
#                     image, steering_angle = augment(image, steering_angle)

#                 # Append the image, steering angle, and labels to the batch
#                 batch_images.append(image)
#                 batch_steering_angles.append(steering_angle)
#                 batch_labels.append(objects)

#                 # Yield a batch
#                 if len(batch_images) == batch_size:
#                     yield np.array(batch_images), np.array(batch_steering_angles), batch_labels
#                     batch_images = []
#                     batch_steering_angles = []
#                     batch_labels = []


def augment(image, steering_angle):
    """Apply random augmentation (e.g., horizontal flip)."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle