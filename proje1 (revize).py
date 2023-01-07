# -*- coding: utf-8 -*-
"""proje1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BIMyPiBayLhLUpuQ6wQHmQD3gC5vUJuh

Gerekli Drive bağlantılarının yapılması:
"""

import pandas as pd
import os

from google.colab import drive
drive.mount('/content/drive')

!ls

os.environ['KAGGLE_CONFIG_DIR']= "/content/drive/MyDrive/Colab_Notebooks/input"

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/Colab_Notebooks/input"

!pwd

!kaggle datasets download -d deadskull7/fer2013

!ls

!unzip \*.zip && rm *.zip

!ls

#Set içeriğindeki ilk 5 eğitim içeriğini gördüm. Veri görselleştirilir.
data=pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/input/fer2013.csv')
print("Number of Labels:", data.emotion.max() + 1)
pd.set_option('max_colwidth',100)
data.head(5)

"""Gerekli paket ve kütüphanelerin kurulumu:"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
# %matplotlib inline

from keras.callbacks import ModelCheckpoint

model = ks.Sequential([
    ks.layers.LSTM(units=1, input_shape=(5, 1))
])

"""**Eğitim ve test performansının ölçüldüğü veri sayılarını inceleyelim.**
Usage kolonunda verisetindeki örnekler kaç gruba ayrıldığını görebiliriz. Kaggleda genelde bu şekilde submit sonrası asıl test edilmesi için verisetinden bir kısmını "PrivateTest" olarak ayırırlar.
privateTest olanlar daha sonra test etmek içindir. Public olanlar ise en sondaki başarım oranını test etmek içindir. 
"""

data["Usage"].value_counts()

np.unique(data["Usage"].values.ravel()) 

print('Eğitim verisetindeki örnek sayısı: %d'%(len(data[data.Usage == "Training"])))

#sadece eğitim örneklerini train_data değişkenine aldık
train_data = data[data.Usage == "Training"]

#eğitim örneklerinin piksel değerleri bize tablo halinde yan yana verildiği için boşluklardan parse ederek liste olarak değişkene aldık. Büyük bir matris oluştu ve her biri içinde pixeller mevcut.
train_pixels = train_data.pixels.str.split(" ").tolist() 
print(len(train_pixels))
train_pixels = pd.DataFrame(train_pixels, dtype=int)
train_images = train_pixels.values
train_images = train_images.astype(np.float64)

print(train_images)
print(train_images.shape)

#Görüntüyü 48x48 piksel şeklinde göstermek için bir fonksiyon tanımlayalım, reshape ve gri seviye belirt. 
def show(img, label="None"):
    show_image = img.reshape(48,48)
    plt.axis('off')
    plt.title(label)
    plt.imshow(show_image, cmap='gray')

#dict ve duygu_siniflari = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# veriseti içindeki eğitim kümesinden indexli bir örnek görseli deneyelim 
label_index = train_data.emotion
index = label_index[30]
label = labels[index]
show(train_images[30], label)

label_index = train_data.emotion
index = label_index[28705]
label = labels[index]
show(train_images[28705], label)

"""Eğitim kümesinde kaç sınıf bulunuyor, görelim."""

train_labels_flat = train_data["emotion"].values.ravel()
train_labels_count = np.unique(train_labels_flat).shape[0]
print('Farklı yüz ifadelerinin (duygu durumları) adedi: %d'%train_labels_count)

"""**One Hot ile eğitim kümesindeki verilerin her birine düşen sınıfı yani eğitim işlemi boyutunu görelim.**"""

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

x_train = dense_to_one_hot(train_labels_flat, train_labels_count)
x_train = x_train.astype(np.uint8)

print(x_train.shape)

"""**TEST VERİSİ ÖN İŞLEME ADIMI**
Eğitim işlemi için veri kümesinde ayrılmış olan PublicTest kısmını alırız.
"""

y_train = dense_to_one_hot(train_labels_flat, train_labels_count)
y_train = y_train.astype(np.uint8)

print(y_train.shape)

np.unique(data["Usage"].values.ravel()) 

print('Test verisetindeki örnek sayısı: %d'%(len(data[data.Usage == "PublicTest"])))

test_data = data[data.Usage == "PublicTest"] 
test_pixels = test_data.pixels.str.split(" ").tolist() 

test_pixels = pd.DataFrame(test_pixels, dtype=int)
test_images = test_pixels.values
test_images = test_images.astype(np.float64)

print(test_images.shape)
#Toplam 3589 görüntü ve her birinde 2304 pixel değeri ile ifade ediliyor.

#eğitim kümesinden bir örenk alıp, test edelim
test_label_index = test_data["emotion"].values
index = test_label_index[3588]
label = labels[index]
show(test_images[3588], label)

"""One Hot ile test kümesindeki verilerin her birine düşen sınıfı yani eğitim işlemi boyutunu görelim.
3589 görüntü ve yine 7 sınıf. test kümesi içinn
"""

test_labels_flat = test_data["emotion"].values.ravel()
test_labels_count = np.unique(test_labels_flat).shape[0]

y_test = dense_to_one_hot(test_labels_flat, test_labels_count)
y_test = y_test.astype(np.uint8)


print(y_test.shape)

"""**TEST KÜMESİNDEN toplu ÖRNEK GÖRÜNTÜLER  **"""

plt.figure(0, figsize=(9,6))
for i in range(1,13):
    plt.subplot(3,4,i)
    plt.axis('off')
    image = test_images[i].reshape(48,48)
    plt.imshow(image, cmap="gray")

plt.tight_layout()
plt.show()

"""# **DERİN EVRİŞİMLİ SİNİR AĞI MODELİ TANIMLANMASI**"""

#sequential ile boş bir hacim tasarlanır.
model=Sequential()
#1. KATMAN 3 filtreden oluşuyor.
model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
#2. KATMAN
model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2)) #%20 unutma işlemi(nöron silme-dropout)
#3. KATMAN
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
#4. KATMAN
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
#5. KATMAN
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2)) #%20 unutma işlemi(nöron silme-dropout)

#6. KATMAN
model.add(Conv2D(128, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
#7. KATMAN
model.add(Conv2D(256, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

# TAM BAĞLANTI KATMANI Flatten komutuyla matris vektör haline çevirilir. Dense ile de tam bağlantı sağlanır.

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
#ÇIKIŞ KATMANI Dense, 7 sınıf sayısı kadar seçilir
model.add(Dense(7))
model.add(Activation('softmax')) #Sınıflama işlemi (7 duygu sınıfı var)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #opmizasyon ve başarım hesaplama metriklerinin belirlenmesi
model.summary() #model özetini görselleştirelim

"""Eğtim ve Test kümelerinin eleman sayısı, yükseklik ve genişlik, kanalsayısı bilgilerini ekrana yazdıralım."""

x_train = train_images.reshape(-1, 48, 48, 1)
x_test = test_images.reshape(-1, 48, 48, 1)

print("Train:", x_train.shape)
print("Test:", x_test.shape)
#eğitim train için 28709 tane 48x48 filtreli ve bir kanallı görüntümüz var

#Kümeler içine elean ve duygu sınıfalrı sayısı
print("Train:", y_train.shape)
print("Test:", y_test.shape)

"""**Eğitim işleminin gerçekleşmesini istediğimiz epoch, batchsize gibi değerlerin belirlenmesi ve eğitim sonucunda ağırlıkların .h5 dosyası olarak kaydedilmesi işlemleri**"""

# en başarılı ağırlıkları kaydet
checkpointer = ModelCheckpoint(filepath='/content/drive/MyDrive/Colab_Notebooks/input/ModelCheckpoint', verbose=1, save_best_only=True)

epochs = 20
batchSize = 256

# modeli çalıştır model.fit çalıştırılacak olan alan
hist = model.fit(x_train, y_train, #model eğitilir
                 epochs=epochs,
                 shuffle=True,
                 batch_size=batchSize, 
                 validation_data=(x_test, y_test),
                 callbacks=[checkpointer], verbose=2)

# save model to json
model_json = model.to_json()
with open("/content/drive/MyDrive/Colab_Notebooks/input/ModelCheckpoint/saved_model.pb", "w") as json_file:
  json_file.write(model_json)

import shutil
model.save("/content/drive/MyDrive/Colab_Notebooks/input/ModelCheckpoint/variables/face_model2.h5")
shutil.make_archive('myy_model', 'zip', '/content/drive/MyDrive/Colab_Notebooks/input/myy_model')

"""**Eğitim sonucu elde edilen Eğitim ve Geçerleme (Validation) sonuçlarının grafiksel olarak ifade edilip ekrarna yazdırılması işlemleri.**"""

# Plot training & validation accuracy values figure subplot plot fonksiyonları kullanılır. 
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Eğitim Başarısı', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist.history['loss'], color='b', label='Training Loss')
plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(hist.history['accuracy'], color='b', label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show() #ekraan göster

#**PrivateTest** örnekleri ile test edelim.
test = data[["emotion", "pixels"]][data["Usage"] == "PrivateTest"]
test["pixels"] = test["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
test.head()

x_test_private = np.vstack(test["pixels"].values)
y_test_private = np.array(test["emotion"])

x_test_private = x_test_private.reshape(-1, 48, 48, 1)
y_test_private = np_utils.to_categorical(y_test_private)
;
x_test_private.shape, y_test_private.shape

score = model.evaluate(x_test_private, y_test_private, verbose=0)
print("PrivateTest üzerinde doğruluk başarımı:", score)

"""Veri kümesindeki eğitim kısmı ile modeli eğitip test için ayırılan veri ile test işlemlerini yaptık. *
Şimdi de veriseti dışındaki Farklı görüntülerle test işlemlerini yapıp sonuçları görselleştirelim
"""

# Commented out IPython magic to ensure Python compatibility.
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

from scipy import ndimage
#from lr_utils import load_dataset

# %matplotlib inline

"""Gerekli kütüphanleri kurup, daha önceki eğitimde kaydettiğimiz modelin hesapladığını öğrenilmiş ağırlık dosyasını kullanıyoruz."""

data = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/input/ModelCheckpoint/data/fer2013.csv')
data.shape

# en iyi ağırlıkları yükle, önceki modelle beraber kullanımı sağlanırr
loaded = keras.models.load_model("/content/drive/MyDrive/Colab_Notebooks/input/ModelCheckpoint/variables/face_model2.h5")

test_image=x_test_private[15]

custom = model.predict(test_image.reshape(-1, 48, 48, 1))

#1 boyutlandır
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2 array halne getir. Veritabanındaki görseller hep dizi (array) şeklindedir.
x = np.array([48, 48], 'float32')
#x = x.reshape([48, 48);
plt.axis('off')
plt.gray()
plt.imshow(test_image.reshape(48,48))

plt.show()

test_image=x_test[30]

custom = model.predict(test_image.reshape(-1, 48, 48, 1))

#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array([48, 48], 'float32')
#x = x.reshape([48, 48);
plt.axis('off')
plt.gray()
plt.imshow(test_image.reshape(48,48))

plt.show()

# Kendi örneklerimizle test işlem adımları

import cv2

x_test_private = test_images[5].reshape(-1, 48, 48, 1)

!ls '/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images'

image_path = "/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/brando.png"
import keras.utils as image
from tensorflow.keras.utils import load_img

test_image_original = image.load_img(image_path) # orjinal renkli görüntü

test_image = image.load_img(image_path, target_size=(48, 48), grayscale=True)
test_data = image.img_to_array(test_image)

test_data = np.expand_dims(test_data, axis=0)
test_data = np.vstack([test_data])

results = model.predict(test_data, batch_size=1)
results

from google.colab.patches import cv2_imshow

# Commented out IPython magic to ensure Python compatibility.
from tensorflow.keras.preprocessing import image

test_image="/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/brando.png"
img = image.load_img(test_image, target_size=(48, 48))
img_array = image.img_to_array(img)

custom = model.predict(img_array.reshape(-1,48,48,1))

#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array([48, 48], 'float32')
#x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()

import matplotlib.image as mpimg 
from matplotlib.pyplot import imshow
# %matplotlib inline
testim = mpimg.imread('/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/brando.png')
imshow(testim)


plt.show("/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/brando.png")

# Commented out IPython magic to ensure Python compatibility.
test_image="/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/kemal_sunal2.jpg"
img = image.load_img(test_image, target_size=(48, 48))
img_array = image.img_to_array(img)

custom = model.predict(img_array.reshape(-1,48,48,1))

#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array([48, 48], 'float32')
#x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()

import matplotlib.image as mpimg 
from matplotlib.pyplot import imshow
# %matplotlib inline
testim = mpimg.imread('/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/kemal_sunal2.jpg')
imshow(testim)


plt.show("/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/kemal_sunal2.jpg")

# Commented out IPython magic to ensure Python compatibility.
test_image="/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/javierbardem.jpg"
img = image.load_img(test_image, target_size=(48, 48))
img_array = image.img_to_array(img)

custom = model.predict(img_array.reshape(-1,48,48,1))

#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array([48, 48], 'float32')
#x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()

import matplotlib.image as mpimg 
from matplotlib.pyplot import imshow
# %matplotlib inline
testim = mpimg.imread('/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/javierbardem.jpg')
imshow(testim)


plt.show("/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/javierbardem.jpg")

# Commented out IPython magic to ensure Python compatibility.
test_image="/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/hababam.jpg"
img = image.load_img(test_image, target_size=(48, 48))
img_array = image.img_to_array(img)

custom = model.predict(img_array.reshape(-1,48,48,1))

#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array([48, 48], 'float32')
#x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()

import matplotlib.image as mpimg 
from matplotlib.pyplot import imshow
# %matplotlib inline
testim = mpimg.imread('/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/hababam.jpg')
imshow(testim)


plt.show("/content/drive/MyDrive/Udemy_DerinOgrenmeyeGiris/Evrisimli_Sinir_Aglari/Duygu_Tanima/images/hababam.jpg")

import zipfile
with zipfile.ZipFile("myy_model.zip","r") as zip_ref:
    zip_ref.extractall("/content/drive/MyDrive/Colab_Notebooks/input/myy_model")

"""**SONUÇLARIN GÖRSELLEŞTİRİLMESİ ADIMLARI**"""

#sınıflarımız 7 adet duygu durumumuz
class_names = ['kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirma', 'dogal']

ind = 0.1+0.6*np.arange(len(class_names))
width = 0.4  #bar genişliği

color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']

# test resmimizi çizdirelim

plt.imshow(test_image_original)
plt.title('Giriş Resmi', fontsize=16)
plt.axis('off')
plt.show()

#sonuçlarımızı renklendirelim

for i in range(len(class_names)):
  plt.bar(ind[i], results[0][i], width, color=color_list[i])

plt.title("Sınıflandırma Sonuçları", fontsize=20)
plt.xlabel("Yüz İfadeleri Kategorisi",fontsize=16)
plt.ylabel("Sınıflandırma Skoru",fontsize=16)
plt.xticks(ind, class_names, rotation=45, fontsize=14)
plt.show()


print("Sınıflandırma sonucu en yüksek oranla:", class_names[np.argmax(results)])

# en yüksek skorlu duyguya karşılık emoji çizdirelim

emojis_img = image.load_img(root + 'images/emojis/%s.png'% str(class_names[np.argmax(results)]))

plt.imshow(emojis_img)
plt.axis('off')
plt.show()