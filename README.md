# IEOR 4720: Deep Learning
Team B1: Tiffany Soebijantoro, Donghao Li

In this project, we seek to identify which patients have Alzheimer’s Disease based on their brain MRI scans, age, and gender data. We will do so using a 3D CNN model and training it on 6,211 data points. Our model should be able to classify a patient’s diagnosis based on their MRI scan, age, and gender. Our code base is in Python, using NumPy and Pandas to help with data preprocessing, Scipy and Nibabel to process MRI scans, and Tensorflow for the CNN model.


# Step 1: Data Preprocessing (Optional)
Using the uploaded input data CSV files: oasis_label_2.csv, oasis_image_data_dates.csv, and NACC_LABELS_CLASSIFICATION.csv, run CNN_InputData_v2.ipynb to obtain the input tensors. Otherwise, the input tensors have also been uploaded and available to download.

# Step 2: Running the model
Using the input tensors provided (eg. train_img.npy), run cnn_brain_mri_v10.ipynb. This file contains the 3D CNN model with a default num_epochs=3, learning_rate = 0.01, and minibatch_size = 50. Alternatively, run cnn_v10.py.
