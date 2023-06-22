# **DERİN ÖĞRENME İLE YÜZ GÖRÜNTÜLERİNDEN DUYGU ANALİZİNİ TESPİT EDEN PROGRAM VE TASARIMI** 
**Proje 1** 
**Meryem Özlem AYDOĞAN**

~~From model content:

**1.1 Purpose of the Project:**

The project of a program that detects emotion analysis from facial images with deep learning is a research topic that can be applied in many computer vision fields such as human�computer interaction and emotional computing. In the project, it is aimed to analyze the emotion of the face in the image specified by the user with deep learning methods. A system model was developed using Convolutional Neural Networks (CNN) structure, which is one of the deep learning algorithms for emotion analysis. This system model was applied to the dataset and maximum accuracy was achieved. Through different experiments, comparison support was provided, prediction rates were measured and the optimal value ranges were determined. The aim of the project is to work on the current research topic with continuous development and to contribute to the development by bringing different perspectives.


**2.PROJECT CONTENT AND SCOPE:**

Recently, face recognition and detection systems have been used in many commercial, military, security, social and are frequently used in psychological applications. Analyses have shown that human faces It involves the identification and interpretation of their movements. Difficult even for humans the ease of testing and determining emotional expressions that can be analyzed in a computer environment that it is thought to provide a new and innovative approach to deep learning has brought popularity to the field. In this context computer vision should also be mentioned. Computer vision is nowadays used for face and emotion classification is widely used in the fields of face recognition. Face recognition, image or automatic identification or verification of persons in data derived from videos process. As a result of the additions made throughout the project scope, the success rate of the revised model and with the help of the graphs drawn that the availability increases in direct proportion. observed. One of the deep learning techniques for feature extraction, artificial neural a new model using Convolutional Neural Networks (ESA-CNN), an approach involving networks have been developed. Commonly used combinations for model training have been tested in recent studies and The effect of the classification algorithms on the performance of the classification algorithms is analyzed. By making evaluations, the classification algorithm that performs well and the real emotion classification from facial images using a time convolutional neural network architecture The project that realizes the process simultaneously has been revealed.

**3.1 Technologies, Platforms and Languages Used in the Project:**

The development of the project application was carried out using deep learning. Application model consists of a total of 9 layered neural networks. The application is based on Google Colaboratory and Designed on Visual Studio Code platforms. Colaboratory, deep learning and machine is a free cloud service where learning models can be developed on GPU. The model training is realized quickly and gives outputs, and the ready-made Python libraries included because it does not require additional downloads and provides easy-to-use support was selected. Keras and TensorFlow Python libraries were used on Colaboratory. The Visual Studio Code platform was used to test the model and to obtain the accuracy rate results. It was preferred to provide a connection to the interface where it will be displayed graphically together with the visual. Creating the model, installing libraries, providing plugin support and interface including the establishment of connections, was developed using the Python programming language.

**3.2 Infrastructure, Hardware and Software Features:**

A high quality dataset is needed for high performance of deep learning. This Therefore, we should look for datasets where training and testing performance is measured to be high and the appropriate set should be selected. In this study, using the fer2013 dataset, we used deep learning with emotion a project has been developed for recognition. The Fer2013 dataset was used in the emotion detection project meets the requirements. The Fer2013 dataset contains a total of 35887 images. Images 28709 for training and 7178 for Public and Private tests. Public tests used to test the success rate after the model is finished, while Private tests are used to test the sets aside some of the images in the set as "PrivateTest" and then uses them to test is used. The technical details of the visuals used for the project are provided by this dataset. can be examined. Thus, it is possible to see how many groups the samples in the data set used in the columns and data visualization can be applied to the fields within the set. The dataset used consists of 35887 rows and 3 columns. In this data set, seven emotions There are pictures aimed at identifying emotions. These emotions are angry (4593), disgust (547) fear (5121), happy (8989), sad (6077), surprised (4002), neutral (6198 pieces). The structure of the images is calculated in 48x48 size and in shades of gray with the help of functions. is organized in such a way that it can be used as a training model. Separate training with the visuals in the model content and the new model developed was tested. In the study carried out with the model, seven different emotion classes (fear, anger, disgust, happiness, neutral, sadness, surprise) is addressed.


**Source codes are discussed in detail in the file contents. It will be updated when necessary improvements are made. You can contact for any questions or revisions you deem necessary in the project. I am always open to improve myself.**

**Kaynak kodlar dosya içeriklerinde detaylıca ele alınmıştır. Gerekli iyileştirmeler yapıldığında güncellenecektir.**
**Herhangi bir sorunuz veya projede gerekli gördüğünüz revizeler için iletişime geçebilirsiniz. Kendimi geliştirmek için her zaman açığım.**



![Ekran görüntüsü_20230108_163812](https://user-images.githubusercontent.com/82104183/211200059-776537e1-7a8b-434c-b2ca-6931b49e005e.png)
![Ekran görüntüsü_20230108_162921](https://user-images.githubusercontent.com/82104183/211200077-a1efb4dd-29cb-43fc-a1d3-ce288e6fc0e8.png)
![Ekran görüntüsü_20230108_163507](https://user-images.githubusercontent.com/82104183/211200079-ca1fc55a-cc8e-424a-b98d-e03713c8e289.png)
![Ekran görüntüsü_20230108_163642](https://user-images.githubusercontent.com/82104183/211200081-581ad6f3-f79c-4a82-be67-8450389c80c6.png)
![Ekran görüntüsü_20230108_162844](https://user-images.githubusercontent.com/82104183/211200082-b7d804eb-306a-401e-b255-b8c52d396fe2.png)

**Arayüz ekranında seçilen görüntünün duygu analiz sonuçları: **


![image](https://user-images.githubusercontent.com/82104183/211200180-d99bce27-ade9-4d91-b4e9-64d2b2da9057.png)
![Ekran görüntüsü_20230108_170047](https://user-images.githubusercontent.com/82104183/211201265-ba8d5e9f-e294-4b21-9841-74e8dbd857ff.png)
![Ekran görüntüsü_20230108_165925](https://user-images.githubusercontent.com/82104183/211201295-ba1fd018-48ef-454e-b2be-44d8eddb26bb.png)
