# **DERİN ÖĞRENME İLE YÜZ GÖRÜNTÜLERİNDEN DUYGU ANALİZİNİ TESPİT EDEN PROGRAM VE TASARIMI** 
**Proje 1** 
**Meryem Özlem AYDOĞAN**

~~Model içeriğinden:

1.1 Projenin Amacı:

Derin öğrenme ile yüz görüntülerinden duygu analizini tespit eden program projesi; insanbilgisayar etkileşimi, duygusal hesaplama gibi birçok bilgisayarlı görü alanında uygulanabilen 
bir araştırma konusudur. Projede, kullanıcı tarafından belirlenen görseldeki yüzün, derin 
öğrenme yöntemleri ile duygu analizinin yapılması amaçlanmıştır. Duygu analizi için derin 
öğrenme algoritmalarında olan Convolutional Neural Networks (CNN) yapısı kullanılarak 
sistem modeli geliştirilmiştir. Oluşturulan bu sistem modeli, veri kümesine uygulanmış ve 
maksimum doğruluk başarısını elde edilmiştir. Yapılan farklı deneyler sayesinde,
karşılaştırma desteği sunulmuş, tahmin oranları ölçülmüş ve optimal olan değer aralıklarının 
belirlenmesi esas alınmıştır.
Projenin amacı, sürekli gelişim göstererek güncel olan araştırma konusu üzerinde çalışmak ve 
gelişime farklı bakış açıları kazandırarak katkıda bulunmaktır.

2.PROJE İÇERİĞİ VE KAPSAMI:

Son zamanlarda yüz tanıma ve algılama sistemleri birçok ticari, askeri, güvenlik, sosyal ve 
psikolojik uygulamalarda sıkça kullanılmaktadır. Yapılan analizler; insan yüzlerinin 
hareketlerinin tanımlanmasını ve yorumlanmasını içermektedir. İnsanlar tarafından bile zor 
analiz edilebilen duygusal ifadeler bilgisayar ortamında test edilip belirlenmesinin kolaylık 
sağlayacağının düşünülmesi derin öğrenme alanına popülerlik kazandırmıştır. Bu bağlamda 
bilgisayarlı görü alanına da değinilmelidir. Bilgisayarlı görü, günümüzde yüz ve duygu 
sınıflandırma alanlarında yaygın olarak kullanılmaktadır. Yüz tanıma, görüntü veya 
videolardan elde edilen verilerdeki kişilerin otomatik olarak tanımlanması veya doğrulanması
işlemidir. Yüz tanıma işlemlerinin dört temel aşaması vardır. Bu işlemler sırasıyla yüz
algılama, normalleştirme, öznitelik çıkarma ve sınıflandırmadır. Normalleştirme ve 
sınıflandırma algoritmaları yüz tanımada ne kadar başarılı olursa olsun, eğer özellik çıkarma 
aşaması başarılı olmazsa o sistem istenilen başarıyı yakalayamamaktadır.
Proje kapsamı boyunca yapılan eklemeler sonucu revize edilen modelin başarı oranı ve 
kullanılabilirliği doğru orantılı şekilde artış gösterdiği çizilen grafikler yardımıyla 
gözlenmiştir. Öznitelik çıkarımı için derin öğrenme tekniklerinden biri olan ve yapay sinir 
ağları içeren bir yaklaşım olan Evrişimli Sinir Ağları (ESA-CNN) kullanılarak yeni bir model 
geliştirilmiştir. 
Model eğitimi için yaygın olarak kullanılan kombinasyonlar son çalışmalarda test edilmiş ve 
sınıflandırma algoritmalarının gösterdikleri başarım sonuçlarına etkisi incelenmiştir. 
Değerlendirmeler yapılarak, iyi performans gösteren sınıflandırma algoritması ve gerçek 
zamanlı evrişimli sinir ağları mimarisi kullanılarak, yüz görsellerinden duygu sınıflandırması 
işlemi eş zamanlı olarak gerçekleştiren proje ortaya çıkarılmıştır.

3.1 Proje Kapsamında Kullanılan Teknolojiler, Platformlar ve Diller:

Proje uygulamasının gelişimi, derin öğrenme yöntemi ile gerçekleştirilmiştir. Uygulama 
modeli ise toplam 9 katmanlı sinir ağından oluşmaktadır. Uygulama, Google Colaboratory ve 
Visual Studio Code platformlarında tasarlanmıştır. Colaboratory, derin öğrenme ve makine 
öğrenimi modellerinin GPU üzerinden geliştirilebildiği ücretsiz bulut servisidir. Modelin 
eğitiminin hızlıca gerçekleştirilip çıktılar vermesi, içinde bulunan hazır Python kütüphaneleri
sebebiyle ek indirmelere gerek duymaması ve kolay kullanım desteği sağladığı için 
seçilmiştir. 
Colaboratory üzerinde ise, Keras ve TensorFlow Python kütüphaneleri kullanılmıştır. 
Visual Studio Code platformu ise modelin test edilip, doğruluk oran sonuçlarının seçilen 
görselle birlikte grafiksel olarak gösterileceği arayüze bağlantı sağlaması için tercih edilmiştir.
Modelin oluşturulması, kütüphanelerin kurulumu, eklenti desteklerinin sağlanması ve arayüz 
bağlantılarının kurulması dahil her işlem Python programlama dili kullanılarak geliştirilmiştir.

3.2 Altyapı, Donanım ve Yazılım Özellikleri:

Derin öğrenmenin yüksek başarımı için kaliteli bir veri setine ihtiyaç duyulmaktadır. Bu 
sebeple, eğitim ve test performansının yüksek olarak ölçüldüğü veri setlerine bakılmalı ve 
uygun olan set seçilmelidir. Çalışmada, fer2013 veri seti kullanılarak derin öğrenme ile duygu 
tanımaya yönelik bir proje geliştirilmiştir. Fer2013 veri seti duygu tespit projesindeki ihtiyacı 
karşılamaktadır. Fer2013 veri setinde toplam 35887 görüntü bulunmaktadır. Görüntülerin 
28709 tanesi eğitim, 7178 tanesi ise Public ve Private testler için ayrılmıştır. Public testler 
model bitirildikten sonraki başarım oranını test etmek için kullanılırken, Private testler ise veri 
setindeki görsellerden bir kısmını "PrivateTest" olarak ayırır ve daha sonra test etmek için 
kullanılır. Proje için kullanılan görsellerin teknik detayları bu veri seti sayesinde
incelenebilmektedir. Böylece kolonlarda kullanılan veri setindeki örneklerin kaç gruba 
ayrıldığı görülebilir ve set içindeki alanlara veri görselleştirilmesi uygulanabilmektedir.
Kullanılan veri seti 35887 satır ve 3 kolondan oluşmaktadır. Bu veri setinde yedi duyguyu 
tespit etmeye yönelik resimler bulunmaktadır. Bu duygular kızgın (4593 tane), iğrenme (547 
tane), korku (5121 tane), mutlu (8989 tane), üzgün (6077 tane), şaşırma (4002 tane), nötr 
(6198 tane) dur. Görsellerin yapısı fonksiyonlar yardımı ile 48x48 boyutunda ve gri tonlarında 
olacak şekilde düzenlenmiştir. Model içeriğindeki görsellerle ayrı ayrı eğitim 
gerçekleştirilerek geliştirilen yeni model test edilmiştir. Model ile gerçekleştirilen çalışmada, 
her bir veri setinde yedi farklı duygu sınıfı (korku, öfke, iğrenme, mutluluk, nötr, üzüntü, 
şaşırma) ele alınmıştır.

**Kaynak kodlar dosya içeriklerinde detaylıca ele alınmıştır.**

![Ekran görüntüsü_20230108_163812](https://user-images.githubusercontent.com/82104183/211200059-776537e1-7a8b-434c-b2ca-6931b49e005e.png)
![Ekran görüntüsü_20230108_162921](https://user-images.githubusercontent.com/82104183/211200077-a1efb4dd-29cb-43fc-a1d3-ce288e6fc0e8.png)
![Ekran görüntüsü_20230108_163507](https://user-images.githubusercontent.com/82104183/211200079-ca1fc55a-cc8e-424a-b98d-e03713c8e289.png)
![Ekran görüntüsü_20230108_163642](https://user-images.githubusercontent.com/82104183/211200081-581ad6f3-f79c-4a82-be67-8450389c80c6.png)
![Ekran görüntüsü_20230108_162844](https://user-images.githubusercontent.com/82104183/211200082-b7d804eb-306a-401e-b255-b8c52d396fe2.png)
**Arayüz ekranında seçilen görüntünün duygu analiz sonuçları: **
![image](https://user-images.githubusercontent.com/82104183/211200180-d99bce27-ade9-4d91-b4e9-64d2b2da9057.png)
![Ekran görüntüsü_20230108_170047](https://user-images.githubusercontent.com/82104183/211201265-ba8d5e9f-e294-4b21-9841-74e8dbd857ff.png)
![Ekran görüntüsü_20230108_165925](https://user-images.githubusercontent.com/82104183/211201295-ba1fd018-48ef-454e-b2be-44d8eddb26bb.png)
