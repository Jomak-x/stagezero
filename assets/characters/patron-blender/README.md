# Patron — Blender karakter modeli

Kullanıcının sağladığı `Gemini_Generated_Image_6vmhe66vmhe66vmh.jpg` referansından Blender 5.2.2 LTS içinde yüzey modelleme yöntemiyle oluşturulmuş stilize karakter. Önceki TripoSR geometrisi kullanılmadı.

## Dosyalar

- `patron.blend`: Düzenlenebilir karakter geometrisi, ayrı malzemeler, subdivision düzenleyicileri, ışıklar ve kamera.
- `patron.glb`: Yalnızca karakter; stüdyo zemini, ışıklar ve kamera içermez. Bağımsız dosya olarak görüntülenebilir.
- `hero.png`, `front.png`, `side.png`: Modelin gerçek Blender renderları.
- `model-info.json`, `validation.json`: Üretim bilgileri ve dosya kontrol sonuçları.
- `source/`: Yeniden üretim için kullanılan Blender Python modülleri.

## Kapsam

Model iri gövdeli, koyu erik renkli takım elbiseli karakterin sadeleştirilmiş yorumudur. Referanstaki yüz, saç ve kumaş ayrıntıları birebir yeniden üretilmemiştir.

Yüz, saç, kıyafet, eller ve ayakkabılar ayrı düzenlenebilir parçalardır. Renkler materyaller ve yüzde vertex color ile tanımlanmıştır; kaplama atlası veya tamamlanmış UV paketi yoktur. Doku dosyası indirmek gerekmez.

Bu sürüm statiktir: iskelet, skin weight, yüz blendshape ve animasyon içermez. Çok parçalı geometrinin dövüş animasyonları için retopoloji, rig ve deformasyon kontrolüne ihtiyacı vardır. ARDY entegrasyonu yapılmamıştır; doğrudan çalışan bir ARDY karakter paketi olarak değerlendirilmemelidir. GLB, ayrıntılı inceleme içindir; oyun performansı için ayrıca optimize edilmelidir.

Blender koordinatları metre, Z yukarı, karakterin önü −Y'dir. GLB dışa aktarımında Y yukarı ve ön +Z olur.

## Yeniden üretim

`source` klasöründe Terminal açın. Çıktı için ayrı bir klasör seçin; komut aynı adlı dosyaların üzerine yazar:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python assemble.py -- --final --size 1100 --extra-views front,side --output-dir ../rebuilt
```

Komut yeni bir arka plan Blender sahnesi oluşturur; açık Blender penceresindeki çalışmayı değiştirmez.
