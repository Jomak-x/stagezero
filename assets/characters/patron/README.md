# Patron — yerel 3B karakter taslağı

Kullanıcının Gemini karakter referansından, 26 Eylül 2026 tarihinde Apple M4 / 16 GB MacBook Air üzerinde TripoSR ile üretildi. Modelleme ve renk üretimi CPU üzerinde, UV atlasının rasterizasyonu yerel OpenGL ile yapıldı.

Bu bir **statik model taslağıdır**. İskelet, skinning ağırlıkları, animasyon ve ARDY entegrasyonu henüz yoktur. Yüz, eller ve giysi yüzeyleri elle temizlenmeye ihtiyaç duyabilir. Üretilen geometri, referansın birebir yeniden yapımı değildir.

| Dosya | İçerik |
|---|---|
| `patron-textured-v1.glb` | İlk UV kaplamalı model; 36.264 üçgen, 1024×1024 gömülü base-color dokusu |
| `patron-v1.glb` | İlk ham TripoSR modeli; vertex renkleri, özgün Z-up koordinatları |
| `patron-basecolor-v1.png` | İlk sürümün dışarı aktarılmış renk atlası |
| `reference-front.png` | Referans sayfasından ayrılan önden görünüş |
| `foreground-front.png` | Yerelde arka planı temizlenen referans |
| `input-front.png` | TripoSR'a verilen, kareye yerleştirilmiş görüntü |
| `generation.json` | İlk üretimin ölçümleri |
| `textured-generation.json` | İlk UV kaplamalı modelin ölçümleri |

Kaplamalı modeller metre ölçeğinde, **+Y yukarı, +Z öne**, ayakları `Y=0` üzerinde olacak şekilde düzenlendi. **1,80 m boy bir prototip tercihidir**; referanstan ölçülmüş gerçek bir boy değildir. GLB, UV ve renk dokusunu kendi içinde taşır; PNG yan dosyası düzenleme kolaylığı içindir. Normal/roughness haritaları üretilmedi; materyalde sabit pürüzlülük kullanıldı.

TripoSR tek görünüşle çalıştığından yalnız ön referans kullanıldı. Sırt ve yan detayları model tarafından tahmin edildi; kaynak sayfadaki yan/arka görünümler 3B üretime girdi olarak verilmedi.

Üretim kaynağı: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR), commit `107cefdc244c39106fa830359024f6a2f1c78871`; ağırlıklar [stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR). Kaynak lisansı `TRIPOSR-LICENSE.txt` içinde korunmuştur.

Yerel kurulum ve tekrar üretim betikleri `.runtime/character3d/` altında bulunur. Bunlar Git tarafından takip edilmeyen makineye özel dosyalardır; temiz bir klonda hazır bulunmaz. `environment.txt` kurulu üretim bağımlılıklarını, `scratch/api/` kullanılan API'lerin sürüm bazlı doğrulamalarını kaydeder. CUDA bağımlı marching-cubes yerine CPU üzerinde scikit-image kullanıldı; eksen eşlemesi asimetrik merkezli bir test yüzeyiyle doğrulandı.

Sonraki aşama: yüz/el/topoloji düzeltmeleri, Core iskeletine uyarlama, skinning ağırlıkları ve dövüş hareketleriyle deformasyon testi. Bu GLB mevcut StageZero G1 görüntüleyicisine doğrudan animasyonlu karakter olarak yüklenemez.
