                                                    **GENDER CLASSIFICATION**

The **Gender Classification** project relies on image-based data to automatically classify the gender of a person by adopting a deep learning paradigm. The system is trained on Image Dataset 1, comprising two classes:

**• Class 0: Female**

**• Class 1: Male**

The dataset is divided into training and testing subsets to be used as input to a **Convolutional Neural Network (CNN)** model. The CNN is constructed to learn facial patterns differentiating male from female.

Aside from Dataset 1, the model is also evaluated on two multiclass test and training sets to compare its robustness under various conditions of images. To confirm **real-world performance**, the trained model is then tested on a real-world dataset of varied facial images that are not visible to it during training.

Upon validation, the model performs with high accuracy in gender detection, able to identify male and female classes well in both structured and unstructured settings. The project illustrates how deep learning, specifically CNNs, can be utilized in real-time gender classification tasks with potential real-world applications in security systems, marketing analysis, and intelligent human-computer interaction.

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



**Face Recognition**

**Face Recognition** project aims to detect and authenticate human faces based on deep learning methods. The system is trained with an image dataset of labelled facial images from several individuals. It employs **face embedding** methods driven by a **Triplet Network** architecture and a **Convolutional Neural Network (CNN**) for efficient facial feature extraction and matching.

The Triplet Network architecture guarantees that the embeddings for an individual are brought closer together whereas embeddings for others are pushed apart in the feature space. This deep metric learning technique offers high accuracy and robustness, even in real-world difficult situations.

In order to confirm the model's performance, extensive testing is conducted with real-world facial datasets having different lighting conditions, expressions, and angles. The model can generalize very well and has high accuracy in properly recognizing and distinguishing faces.

This method makes it possible to use robust face recognition applicable in applications like surveillance, biometric authentication, intelligent access control systems, and personalized services.

![image](https://github.com/user-attachments/assets/0e8a5f46-bb14-417c-8f0c-b19e148ff26b)

TASK B custom dataset

![image](https://github.com/user-attachments/assets/d49c441d-6b79-4088-8e9e-790ac643aad8)

Positive match (label=1). Best matching identity  
 folder: test-dataset with similarity 0.9908



![image](https://github.com/user-attachments/assets/601fd30c-525a-418f-ae2b-4259ba268bc5)

