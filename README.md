Requirements
Make sure to install the following Python packages:

       pip install torch torchvision scikit-learn pillow
Or use the provided requirements.txt:

      torch>=1.9.0
      torchvision>=0.10.0
      scikit-learn
      pillow
  
  
  
  
  **GENDER CLASSIFICATION**

The **Gender Classification** project relies on image-based data to automatically classify the gender of a person by adopting a deep learning paradigm. The system is trained on Image Dataset 1, comprising two classes:

**• Class 0: Female**

**• Class 1: Male**

The dataset is divided into training and testing subsets to be used as input to a **Convolutional Neural Network (CNN)** model. The CNN is constructed to learn facial patterns differentiating male from female.

Aside from Dataset 1, the model is also evaluated on two multiclass test and training sets to compare its robustness under various conditions of images. To confirm **real-world performance**, the trained model is then tested on a real-world dataset of varied facial images that are not visible to it during training.

Upon validation, the model performs with high accuracy in gender detection, able to identify male and female classes well in both structured and unstructured settings. The project illustrates how deep learning, specifically CNNs, can be utilized in real-time gender classification tasks with potential real-world applications in security systems, marketing analysis, and intelligent human-computer interaction.


The Folder structure look like this,



   ![WhatsApp Image 2025-07-11 at 00 12 46_5aea47db](https://github.com/user-attachments/assets/39b18623-5187-478b-aa1c-e6b5a3f9f24a)


After running the training script,the result


  ![WhatsApp Image 2025-07-04 at 22 52 46_6e8577d6](https://github.com/user-attachments/assets/353ae488-711e-4463-a102-f31cc6bd76e9)



  ![image](https://github.com/user-attachments/assets/97511606-4b53-4390-a1fb-ace92edd0a40)

  Total images processed: 1
  Number of females predicted: 0
  Number of males predicted: 1
  Average confidence for female predictions: 0.00%
  Average confidence for male predictions: 99.93%


  ![image](https://github.com/user-attachments/assets/7fef035b-447b-4028-928f-b8a6d12533de)

  Total images processed: 1
  Number of females predicted: 1
  Number of males predicted: 0
  Average confidence for female predictions: 99.97%
  Average confidence for male predictions: 0.00%



After runnig the evaluation script (evaluate_gender.py), the results



 ![test a](https://github.com/user-attachments/assets/45d712f8-ac9f-4246-a2f0-bec3d8d39d58)




Use the evaluate_gender.py script as test script, just change the test dataset file replacing the val directory path to determine all the parameters like Accuracy,F1 ,Recall,Precision. Test dataset folder structure must maintain with labels male and female folder and each folder contains images of respective gender. 
Run the script

              python evaluate_gender.py


**Face Recognition**

**Face Recognition** project aims to detect and authenticate human faces based on deep learning methods. The system is trained with an image dataset of labelled facial images from several individuals. It employs **face embedding** methods driven by a **Triplet Network** architecture and a **Convolutional Neural Network (CNN**) for efficient facial feature extraction and matching.

The Triplet Network architecture guarantees that the embeddings for an individual are brought closer together whereas embeddings for others are pushed apart in the feature space. This deep metric learning technique offers high accuracy and robustness, even in real-world difficult situations.

In order to confirm the model's performance, extensive testing is conducted with real-world facial datasets having different lighting conditions, expressions, and angles. The model can generalize very well and has high accuracy in properly recognizing and distinguishing faces.


The folder structure look like this


   ![WhatsApp Image 2025-07-11 at 00 13 00_f965716e](https://github.com/user-attachments/assets/b79b3bf4-00c0-4878-b746-ca8ffddf22e6)




This method makes it possible to use robust face recognition applicable in applications like surveillance, biometric authentication, intelligent access control systems, and personalized services.

![image](https://github.com/user-attachments/assets/0e8a5f46-bb14-417c-8f0c-b19e148ff26b)

TASK B custom dataset

![image](https://github.com/user-attachments/assets/d49c441d-6b79-4088-8e9e-790ac643aad8)


Test image



Run the script test_verification.py,
              
              python test_verification.py <test_image_dir> <identity_dir> <trained_model_path>

Positive match (label=1). Best matching identity  
 folder: test-dataset with similarity 0.9908




After running the evaluation script(evaluate_siamese.py),just change the Train and Val directory path.
   This script can be run for test purpose on test dataset ,make sure the test dataset folder structure is like same as Train and Val folder structure.    
       
       
       python evaluate_siamese.py



the result is :


   ![WhatsApp Image 2025-07-04 at 22 55 48_cd9e2738](https://github.com/user-attachments/assets/48cc207a-82db-48dd-b691-1cf029151cb9)
   ![WhatsApp Image 2025-07-04 at 22 55 21_f7795793](https://github.com/user-attachments/assets/f284a617-e894-4d5c-878c-f12db7eab8ed)






![image](https://github.com/user-attachments/assets/601fd30c-525a-418f-ae2b-4259ba268bc5)

