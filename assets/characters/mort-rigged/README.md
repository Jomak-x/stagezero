# Mort: yerel humanoid rig

`mort-rigged.glb`, Studio'nun mevcut Mixamo profiliyle hareket aktarımı alır. `mort-rigged.blend`, Blender 5.2.2 LTS ile hazırlanmış düzenlenebilir kaynaktır; kaplamalar dosyada paketlidir. Orijinal `mort_LP.glb` ve Studio kataloğundaki kaynak değiştirilmedi.

## Dosyalar

- `mort-rigged.glb`: tek skin, 21 kemik, en fazla 3 sıfırdan farklı kemik ağırlığı; gömülü üç 2048 × 2048 kaplama.
- `mort-rigged.blend`: `MortRig` ve `MortMesh`, özgün asimetrik bağlama pozu, düzenlenebilir ağırlıklar. Kamera ve ışıklar inceleme içindir; GLB'ye aktarılmaz.
- `qa-front.png`, `qa-side.png`: ayakta duruş, kol kaldırma, yürüme adımı ve çömelmenin son dosyadan üretilen ön/yan görsel kontrolleri.
- `validation.json`: gerçek uygulama doğrulayıcısı/retargeter ve deforme olmuş geometri ölçümleri.
- `preservation.json`: kaynak/çıktı özeti, geometri ve kaplama karşılaştırması, SHA-256 kayıtları.

## Yapılan hazırlık

Ön, yan ve arka görünümler incelendi. Kaynak T-pozu değildir: sol dirsek bükülü ve el yukarıdadır; sağ el bir aksesuar tutar. Kemikler bu geometriye yerleştirildi. Sağ kolun derinliği, ten yüzeyinin yatay kesitlerinden ayrıca ölçüldü. Bağlama pozu yapay bir T-pozuna çevrilmedi.

İskelet `Hips → Spine → Spine1 → Neck → Head`, iki omuz/üst kol/ön kol/el zinciri ve iki uyluk/baldır/ayak/parmak tabanı zinciri içerir. Zorunlu 12 rolün yanında her iki el ve baş da mevcut Mixamo adlarıyla eşleşir. El uçları, bükülü kaynak kolun doğru kalibrasyonu için özellikle korunur.

Blender otomatik ısı ağırlıkları başlangıç olarak uygulandı; ağırlıksız köşe kalmadı. Ardından kaynak mesh'in 58 ayrı indeksli yüzey bölgesi elle tanımlanan anatomik gruplara ayrıldı. Eldiven, parmak şekilleri, baş ve sağ eldeki aksesuar kendi kemiğine sabitlenir. Dirsek, omuz, kalça ve diz geçişleri düzeltilmiştir. Kolun gövdeye yalnızca temas ettiği malzeme sınırları birleştirilmez; bu, yanlış ağırlık yayılmasını önler. Buna karşılık gerçekten dikili gövde–kol ağzı ve bel sınırlarında ağırlıklar eşleştirilir; kol ağzında yumuşak geçiş çözülerek giyside açıklık oluşması engellenir. Özgün köşeler, üçgenler, UV'ler ve kaplama dosyaları korunur.

Kaynak alt noktası glTF Y ekseninde `−0.5547341108` idi. Geometri ve iskelet birlikte `+0.5547341108` taşındı; yeniden ölçekleme yapılmadı. Özgün yükseklik `1.1117884517` birimdir. Aktarımdan sonraki köşe farkı en fazla `5.96 × 10⁻⁸` birimdir. Üç kaplamanın hem dosya baytları hem açılmış pikselleri kaynakla birebir aynıdır; PBR malzeme tanımı aynıdır.

## Doğrulama ve kullanım sınırları

GLB, `character_assets.inspect_glb` ve `retargeting.build_retargeter` ile geçer; profil `mixamo`, eşleştirme uyarısı yoktur. Tek bağlı kemik hiyerarşisi, geçerli ağırlıklar ve uygun dönüşümler içerir. Gerekli olmayan glTF uzantıları eklenmez.

Görseller uygulamadaki `character_diagnostics.standing_reference` ve kurulu G1 iskeletinin FK çıktıları kullanılarak üretildi. Pozlar üretim `build_retargeter` yolundan geçirildi; oluşan glTF dünya matrisleri Blender'daki aynı skin'e uygulandı. Bunlar yerel sentetik hareket kontrolleridir; ağ üzerinden hareket üretimi yapılmadı. Gerçek Studio yükleme/yeniden bağlanma kontrolü uygulama çalışmasının parçasıdır.

Bağlama pozunda taban `Y ≈ −1.21 × 10⁻⁸`, yani sayısal hassasiyet içinde sıfırdır. Uygulamanın ayakta duruş referansı için gereken tek seferlik taşıyıcı ofseti `−0.0004758056` birimdir. Bu sabit ofsetle ayakta duruş ve kol kaldırmada taban sıfır; seçilen yürüme adımında yaklaşık `+0.00605`, çömelmede `+0.00138` birimdir. Her karede yeniden zemine yapıştırma uygulanmaz; küçük temas farkları korunur.

Kol, el ve aksesuar çevresindeki uzun üçgen gerilmeleri giderildi. Model stilize, kısa uzuvlu ve başlangıçta poz verilmiş bir heykeldir; büyük dirsek/diz bükülmeleri hacim kaybı veya giysi kıvrımı gösterebilir. Bu rig kusursuz deformasyon, ayak kilidi, IK, vücut teması, denge, bağımsız parmak ya da yüz animasyonu garantisi vermez. Yeni hareketlerde özellikle omuz altı, dirsek, diz ve aksesuarın gövdeyle teması gözle kontrol edilmelidir.

## Başka iskeletsiz modeller için aynı yerel süreç

1. Kaynağı koruyup ayrı `.blend` kopyasında ön/yan/arka görünüş ve gerçek ölçeği inceleyin.
2. Kalça, omurga, baş, omuz, dirsek, el, diz ve ayak merkezlerini gerçek geometriye yerleştirin; Mixamo adları ve tek humanoid hiyerarşi kullanın. Bükülü kollarda el kemiği eşleştirmesini ekleyin.
3. Otomatik ağırlıklandırmadan sonra ten, giysi ve aksesuar bölgelerini kontrol edin. Baş/eldeki nesneleri uygun kemiğe bağlayın; omuz, kalça, diz ve dirsekte poz testleriyle ağırlıkları düzeltin.
4. Ölçeği değiştirmeden ayak tabanını zemine taşıyın. Skin, kemikler, UV ve PNG/JPEG kaplamaları gömülü GLB olarak aktarın. Blender'ın bu içe aktarma yolu için `Keep Original` seçeneği yerine gömülü görselleri dışa aktarın; en fazla 4 etki kullanın.
5. Studio uyumluluk kontrolünü ve gerçek hareket aktarımıyla ayakta duruş, kol kaldırma, yürüme ve çömelme görsellerini tekrar doğrulayın. Teknik uyumluluğu deformasyon kalitesiyle karıştırmayın.

Yerel üretim/probe betikleri ve ara görseller, bu çalışma makinesindeki `scratch/character-fix-01a0dd64/rig/` altındadır. Son ağırlık ataması `2026-09-26_150000_component_weights.py`, uygulama poz kontrolü `2026-09-26_143200_validate_retarget.py` ile kaydedilmiştir.
