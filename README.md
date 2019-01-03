# IEOR 4720: Deep Learning
Team B1: Tiffany Soebijantoro, Donghao Li

In this project, we seek to identify which patients have Alzheimer’s Disease based on their brain MRI scans, age, and gender data. We will do so using a 3D CNN model and training it on 6,211 data points. Our model should be able to classify a patient’s diagnosis based on their MRI scan, age, and gender. Our code base is in Python, using NumPy and Pandas to help with data preprocessing, Scipy and Nibabel to process MRI scans, and Tensorflow for the CNN model.


# Step 1: Data Preprocessing (Optional)
Instructions:
Using the uploaded input data CSV files: oasis_label_2.csv, oasis_image_data_dates.csv, and NACC_LABELS_CLASSIFICATION.csv, run CNN_InputData_v2.ipynb to obtain the input tensors. Otherwise, the input tensors have also been uploaded and available to download.

Steps:
1. OASIS - Map MRI scan file names we have to the label from the oasis_label_2.csv file provided.
2. OASIS - Eliminate matches more than 180 days between diagnosis and MRI scan
3. OASIS - Keep matches belonging to the 3 categories: Cognitively normal, AD dementia, and Uncertain dementia.
4. NACC - Map MRI scan file names to labels from data_nacc_diagnosis.xlsx
5. Normalize age to be between 0 and 1
6. Enforce one-hot encoding for sex
7. Merge OASIS & NACC data samples (total: 7768)
8. Divide data samples into training, validation, and test sets (80%, 10%, 10%)
9. Return all samples into 4 NumPy arrays: img (just image path to save memory), age, sex, label

Size of data sets: training - 6211, validation - 773, test - 784

# Step 2: Running the model
Instructions:
Using the input tensors provided (eg. train_img.npy), run cnn_brain_mri_v10.ipynb. This file contains the 3D CNN model with a default num_epochs=3, learning_rate = 0.01, and minibatch_size = 50. Alternatively, run CNN_v10.py.

Steps:
1. Load all input tensors
2. Enforce one-hot encoding for labels
3. Add new axis to img, age, sex
4. Define function to retrieve images from img_path, including downsampling and normalizing images
5. Model
    *  create_placeholders function is to create the placeholders for dataset and labels with the same size.
    *  initialize_parameters function helps define our parameters based on the model architecture.
    *  forward_propagation function is the main part of building model architecture.
    *  compute_cost function helps define what kind of cost we are going to use.
    *  random_mini_batches function is used to divide whole data into batches to increase training speed.
    *  model function is to use all functions above to train the model and output the result. Tune hyperparamers and call model function to get the cost & accuracy rate 
6. Call the model function, specifying learning_rate, num_epochs, minibatch_size, and whether the model is pretrained, to get cost and training & validation accuracy rate
7. Save accuracy rates, epoch_traccuracy and epoch_vlaccuracy, as NumPy arrays (.npy files) if running on Jupyter notebook, or save print output if running Python script.
