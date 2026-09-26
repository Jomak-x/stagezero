# StageZero — UI ve kullanıcı deneyimi iyileştirme planı

Tarih: 26 Eylül 2026. Durum: ilk UI/UX paketi uygulandı; aşağıdaki diğer aşamalar sırada.

## Uygulanan ilk paket

Kullanıcının devam isteği üzerine yerel `/Users/leoy/Code/ShellHacks` checkout'u yeni `codex/studio-ux` branch'inde GitHub main `bad21449ad70a613ec06f180e9379b2252adddfd` sürümüne fast-forward pull ile güncellendi. Bu main sürümü Welcome, yerel sahne üretimi ve GLB çalışmalarını içeriyor. Önceden izlenmeyen yerel GLB planı, upstream dosyası gelmeden `.runtime/studio-ux-01a0dd1a/GLB-INTEGRATION-PLAN.local-before-pull.md` altında korundu.

- Motion boş durumunda ilk hareket açıklaması ve açık Wave / Walk / Dance örnekleri, komut alanından önce gösteriliyor.
- “Save action” yerine “Update motion” kullanılıyor; yeniden üretilecek sonraki aksiyon sayısı açıklanıyor.
- Üretim formunda gerçek controller ilerlemesi ve geçen süre gösteriliyor. Retry yalnızca aynı taslak ve aynı kaynak sürüm için sunuluyor.
- Proje adı, tek Kaydet düğmesi ve gerçek kayıt durumu sekmelerin üstünde. Başarılı disk kaydı ile başarısız tarayıcı indirme bildirimi birbirinden ayrılıyor.
- Mobil panel kapalı başlıyor; klavyeyle açılıyor ve ölçülen timeline alanının üstünde kalıyor. Kapalıyken gelen kontrollerin boyutları korunuyor.
- Timeline komutları dar ekranda satıra sarılıyor; dokunma hedefleri 44px ve klavye odağı görünür.

Doğrulama: 254 Python unittest, 14 frontend testi, TypeScript/Vite build ve diff kontrolü başarılı. Bağımsız Astra/high incelemesinde kalan bulgu yok. Gerçek tarayıcıda 390×844 ilk açılış ve 1280×720 masaüstü kontrol edildi; iki sıralı gerçek wave isteği 104'er kare üretti, Update motion/Undo ve disk kaydı/yeniden açma çalıştı. İndirme bağlantısı arayüzde oluştu; in-app browser download-event yakalama zaman aşımına uğradığından tarayıcı indirme tamamlanması doğrulanmış sayılmadı. Üretim hatası, retry, statik GLB ve kayıt durumu davranışları odaklı testlerle doğrulandı.

Yerel önizleme: `http://127.0.0.1:2345/`. Değişiklikler henüz commit/push edilmedi. Mevcut diğer servisler korunuyor. Sahne undo, ana sekmelerin yeniden düzenlenmesi ve timeline ruler için dokunmatik scrubbing bu pakete dahil değil.

## İlk planlama turunda incelenen sürüm ve kapsam

- GitHub: [Jomak-x/stagezero](https://github.com/Jomak-x/stagezero), `main`, [`fdd9e33`](https://github.com/Jomak-x/stagezero/tree/fdd9e33190195a7a99176fa14697e639b3e22270).
- Yerel checkout `codex/onboarding-controller-qa`, `3f497b1`; incelenen main bu commit'in 9 commit ilerisinde. Kaynak kod GitHub'dan ayrı geçici dizine indirildi; mevcut checkout'a pull/fetch, branch değişimi veya uygulama kodu değişikliği yapılmadı.
- [Bu commit'in CI çalışması başarılı](https://github.com/Jomak-x/stagezero/actions/runs/36232173756). Bu incelemede uygulama çalıştırılmadı ve tarayıcı testi yapılmadı; yerleşim sorunları koddan çıkarılan, uygulama aşamasında doğrulanacak bulgulardır.
- [Açık PR #4](https://github.com/Jomak-x/stagezero/pull/4) ayrı bir gerçek zamanlı viewer/etkileşim çalışması içeriyor. Bu planda mevcut main'in özelliği sayılmadı.
- Varsayılan hedef: ilk kez kullanan bir kişinin masaüstünde hızlıca hareket oluşturup düzenleyebilmesi. Dar ekran ve klavye kullanımı da doğrulama kapsamına alınacak.

## Temel karar

Mevcut React + Mantine + Three.js istemcisini ve Python/Viser kontrol altyapısını koruyarak ekran hiyerarşisini sadeleştirelim. Öncelik, kullanıcının sıradaki adımı ve bir işlemin sonucunu anlaması.

Mevcut güçlü özellikler korunacak: bağımsız kayan sağ panel, altta timeline ve playback, timeline klibinden aksiyon düzenleme/ekleme, örnek komutlar, süre önizlemesi, sürüm kopyalama, aksiyon düzenleme ve sürüm silme için mevcut tek adımlık geri alma, proje açarken yedekleme ve kliple klavye etkileşimi.

Hedef akış: **Yeni çalışma veya örnek seç → hareketi tarif et → oluştur → izle → seçili hareketi düzenle/yenisini ekle → projeyi kaydet.** Sahne seçimi bu akışta isteğe bağlı; ilk hareket için zorunlu kurulum adımı olmayacak.

## Koddan çıkan fırsatlar

| Bulgu | Mevcut davranış ve kanıt | Öneri |
| --- | --- | --- |
| İlk ekran çok fazla kategori sunuyor | `studio_ui.py:81–169`: Motion, Takes, Scene, View, Project ve Guide; örnek komutlar kapalı klasörde. | Boş durumda görünür üç örnek ve tek ana oluşturma düğmesi. Ana panelde Motion / Scene / Takes; Project üst alana, View viewport araçlarına, Guide yardım düğmesine taşınsın. |
| Kaydetme ile yeniden üretme birbirine karışabiliyor | `studio_ui.py:257–269`: “Save action” model üretimi başlatıyor; sonraki hareketlerin yeniden üretileceği küçük notta açıklanıyor. | “Update motion” gibi sonucu anlatan bir etiket; etkilenen sonraki klipleri vurgulama ve işlem öncesi kısa özet. Proje kaydetme ayrı ve tutarlı kalsın. |
| Üretim bilgisi teknik ve dağınık | `directing.py:388,520,589`: parça sayısı zaten üretiliyor; `studio_ui.py:790–798`: üstte küçük metinle gösteriliyor. | Mevcut gerçek ilerlemeyi okunabilir bir durum kartına taşıma; geçen süre, tamamlanan parça sayısı, iptal ve hata sonrası tekrar deneme. Ölçülmemiş ETA veya sahte yüzde gösterilmesin. |
| Kayıt durumu ana çalışma sırasında görünmüyor | `directing.py:693–696` kayıt/snapshot durumunu biliyor; `studio_ui.py:861` bunu Project içeriğine yazıyor. | Üstte proje adı, Kaydedildi/Kaydedilmemiş değişiklikler durumu ve erişilebilir Kaydet düğmesi. Mevcut revision ve snapshot davranışını koruma. |
| Sahne değiştirme kolay, geri dönüş zor | `object_controls.py:11–20,87–105,170–172`: üretim mevcut sahneyi değiştiriyor; sahneye özel undo yok. | Önce sahne snapshot + tek adımlık geri alma; sonra hazır sahne kartları ve değiştirmeden önce sonuç özeti/önizleme. |
| Mobil panel timeline ile aynı alanı kullanabiliyor | `ControlPanel/BottomPanel.tsx:18–49`: başlangıçta açık, alta sabit, 20em genişlik ve %60 azami yükseklik; `App.tsx:382–394`: timeline ayrıca altta. | Dar ekranda tam genişlikte açılır panel; playback ve timeline için korunmuş alan. Klavyeyle açılan/kapanan gerçek düğme. |
| Küçük hedefler ve ruler erişimi | `Timeline.tsx:3056–3074`: 24–25px düğmeler; `3136–3150`: ruler mouse olayları. Klipler ayrıca erişilebilir HTML düğmeleri kullanıyor. | Dokunmada yaklaşık 44px hedefler, belirgin focus, pointer/touch scrubbing ve erişilebilir zaman girişi. Mevcut klip düğmelerini koruma. |
| Bağlantı bilgisi eksik anlam taşıyor | `ControlPanel/ControlPanel.tsx:162–195`: WebSocket bağlantısı göstergesi; AI üretim servisinin hazır olduğu anlamına gelmiyor. | Viewer bağlantısı ile hareket üretiminin kullanılabilirliğini ayırt eden sade metin; bağlantı kopunca eldeki hareketi izleme ve kurtarma yolu. |

Dosya yolları incelenen GitHub commit'ine göredir; bu yerel eski checkout'ta tüm dosyalar henüz bulunmayabilir.

## Önerilen ekran düzeni

- **Üst alan:** proje adı, kayıt durumu, Kaydet, Aç ve yardım. Paylaşılan oturum bilgisi burada kısa biçimde görünür.
- **Orta alan:** büyük 3D sahne; Pan/Orbit/Look, karaktere odaklan ve kamerayı sıfırla araçları mevcut davranışlarını korur.
- **Sağ panel:** Motion / Scene / Takes. İçerik seçime göre değişir; gereksiz ayarlar Advanced altında kalır.
- **Alt alan:** tek playback grubu, zaman göstergesi ve timeline. Seçili klipte “Öncesine ekle”, “Sonrasına ekle”, “Hareketi güncelle”.
- **Boş durum:** “İlk hareketini oluştur” açıklaması; Wave / Walk / Dance örnekleri, komut ve süre. Örnek tıklaması komutu doldurur; üretim ana düğmeyle başlar.
- **Düzenleme durumu:** “2. hareketi güncelliyorsun; sonraki 3 hareket de yeniden üretilecek.” Etkilenen aralık timeline'da görünür; mevcut Undo edit korunur.
- **Sahne düzenleme:** hazır sahne seçimi önde; AI sağlayıcısı, seed ve koordinatlar Advanced altında. Sağlayıcı bulunamıyorsa kullanılabilir seçenek ve neden gösterilir.

Arayüzün mevcut İngilizce dilini ilk pakette tutarlı hale getirelim. Türkçe metinler bu planın açıklamasıdır; ürün çevirisi ayrı bir karar gerektirir.

Sonraki iyileştirmeler: sahnedeki objeyi tıklayıp seçme ve sürükleyerek taşıma; mevcut koordinat alanlarını hassas düzenleme için koruma. İndirme seçeneklerinde “Editable project” ile “Scene JSON” ayrımını açık gösterme. Video çıktısı mevcut main akışında yok; ayrı render/export geliştirmesi olarak ele alınmalı.

## Uygulama sırası

### 0 — Güncel tabanı ve ayrı işleri birleştirmeye hazırlık

1. Uygulamaya başlarken GitHub SHA ve PR durumunu yeniden kontrol et.
2. Paylaşılan sahiplik kaydı ve worker claim'lerini yeniden oku. Welcome, yerel sahne üretimi ve GLB çalışmalarının gerçek diff/commit durumunu çıkar; mevcut main'e dahil olduklarını varsayma.
3. Uygun boş bir yönetilen worktree kullan veya çakışma varsa izole worktree oluştur. Başka çalışanın checkout'unda branch/index değiştirme.
4. Welcome geçişini yeni kompakt App düzenine, panel ölçümleri ve açık sekme korunarak entegre et. Karakter yükleme entegrasyonu UX paketinin zorunlu önkoşulu olmasın.
5. Mevcut test/build ve temel ekran davranışını taban olarak kaydet. Canlı servis değişimi gerekiyorsa kayıttaki sahibiyle ayrı koordine et.

### 1 — İlk kullanım ve anlaşılır işlemler

Boş durum ve örnekler; Motion formunun sadeleşmesi; anlaşılır ana düğmeler; seçili aksiyon ve etkilenen sonraki hareketlerin özeti; gerçek üretim ilerlemesi ve tekrar deneme. Önce `studio_ui.py`, `studio_guide.py` ve gerekli timeline görünümünü değiştir.

Başarı ölçütü: yeni kullanıcı rehber sekmesine gitmeden bir örnek seçip tek üretim düğmesiyle ilk hareketi oluşturabilsin. Kullanıcı testinde hedef, beş ilk kullanıcının en az dördünün bu görevi yardım almadan tamamlaması; model bekleme süresi ayrı ölçülsün.

### 2 — Proje ve sahne güveni

Görünür kayıt durumu ve proje işlemleri; sahne değişikliklerinde snapshot/undo; sahne üretimi devam ederken başka düzenlemelerin sonucunu geç gelen yanıtın ezmemesi. Mevcut otomatik yedek ile kullanıcı tarafından kaydedilmiş projeyi doğru etiketle; sürekli autosave varmış gibi sunma.

Başarı ölçütü: başarısız/iptal edilen işlem önceki çalışmayı korusun; sahne değişimi tek adımda geri alınabilsin; üstteki kayıt durumu gerçekten kaydedilen revision ile uyumlu olsun.

### 3 — Ekran düzeni ve görsel tutarlılık

Sekmeleri sadeleştir; üst proje alanını ve viewport araçlarını düzenle. Ortak yazı boyutları, boşluk, düğme ve durum stilleri kullan. Dar ekran paneli/timeline ilişkisini ve dokunmatik hedefleri düzelt. Kalıcı durumun React'e taşınması gerekiyorsa Viser'daki doğrulanmış state üzerinden açık mesaj alanları ekle; görünen metinleri parse ederek durum türetme.

Başarı ölçütü: 1440×900, 1280×720, 768×1024 ve 390×844 ekranlarda ana işlemler erişilebilir olsun; inspector timeline'ı kullanılamaz hale getirmesin. Klavye ile örnek seçme, oluşturma, klip seçme, zaman değiştirme ve kaydetme tamamlanabilsin.

## Doğrulama ve iş paylaşımı

- İstemci: TypeScript/build ve mevcut Node testleri; yeni davranışlar için odaklı testler. Statik renk/boşluk değişikliklerini taklit eden testler eklenmesin.
- Controller: mevcut Python regresyonları; iptal/hata sonrası çalışma korunması, geç sonuçların reddi, sahne undo ve kayıt revision davranışları.
- Tarayıcı: boş proje → oluştur → klip düzenle → geri al → sahne değiştir/geri al → kaydet → aç. Ayrıca kopan bağlantı, dar ekran, klavye ve varsa Welcome → Studio dönüşü.
- Model hareket kalitesiyle UI doğruluğu ayrı raporlansın. Bu plan AI hareket doğruluğu veya gerçek zamanlı üretim iyileştirmesi vaat etmez.
- Uygulama sırasında bir frontend worker ve bir controller worker dosya sınırları net tutularak paralel çalışabilir; ana ajan entegrasyon ve son tarayıcı doğrulamasını yapar. Paylaşılan `studio_ui.py` için tek sahip belirlenir.
- İlk teslim paketi: aşama 1 ve aşama 2'nin görünür kayıt durumu. İkinci paket: sahne undo/üretim tutarlılığı. Üçüncü paket: ekran yeniden düzenleme, görsel tutarlılık ve dar ekran erişilebilirliği.

## İlk planlama turunda yapılanlar

GitHub main kaynağı ve açık PR kapsamı okundu; iki bağımsız read-only alt ajan UI ve controller akışlarını inceledi. Bu plan dosyası ve kendi koordinasyon claim'i dışında paylaşılan çalışma değiştirilmedi. Pull, merge, commit, paket kurulumu veya canlı uygulama değişikliği yapılmadı.

## Uygulanan ikinci paket — yüzen Studio düzeni

Kullanıcının 26 Eylül görsel yönlendirmesi: Apple benzeri, kenarlara yapışmayan yuvarlatılmış inspector; sahne üzerinde katman; ekranın tüm alt genişliğini kullanan timeline.

- Sahne tam genişlikte; inspector 22 px içeride, 22 px köşe, bulanık yarı saydam grafit yüzey ve yumuşak gölgeyle sahnenin üstünde. Yüksekliği 720 px ile sınırlı, içerik bağımsız kayıyor.
- Timeline ayrı tam genişlik satırında; çevresinde 12 px boşluk, 18 px köşeler. Gruplanmış playback araçları, sade zaman sayacı ve mint seçim vurgusu.
- Sekmeler tek satırda kompakt segmentler; sakin mint ana eylem, ikincil butonlar ve yuvarlatılmış alanlar. Advanced başlıkları ayrı ince kartlar.
- Inspector kapandığında küçük bir kapsüle dönüşüyor; ölçüm için kontrol ağacı bağlı kalıyor. Mobilde panel, ölçülen timeline'ın üzerinde iki yandan 16 px içeride açılıyor.
- Canlı2345 oturumu korunuyor: Viser dosyaları bellekte önbelleklediğinden yeni arayüz için `studio-floating.html?websocket=ws://127.0.0.1:2345` statik giriş kullanılıyor. Normal giriş, sonraki planlı sunucu başlangıcında yeni build'i alacak. Backend yeniden başlatılmadı.
- Doğrulama: 14 frontend testi, TypeScript/Vite build ve bağımsız Astra/high inceleme tamamlandı. Kamera/Welcome çakışması giderildi. 1440 px ve1280 px masaüstü,390 px mobil görünüm; panel kapatma/açma ve klavye erişimi tarayıcıda kontrol edildi. Mobil ölçümde panel altı581 px, timeline üstü593 px; yatay taşma yok. Take2/8.32s korundu. Görseller özel `.runtime/studio-ux-01a0dd1a/floating-*.png` altında.

### Navigation follow-up

The user requested two navigation rows to reduce crowding. The Studio tab strip now groups Motion / Takes / Scene / Character under **Generative**, and View / Project / Guide under **Controller**. The underlying single tablist and original tab values preserve keyboard navigation and server-directed selection; non-Studio tab groups remain flat. The optional Character tab is supported.
- Follow-up correction: Generative and Controller now use two independent rounded bars with open space between them; removed the common enclosing box and divider. CSS-only preview `studio-rows.html` loads the updated source stylesheet directly on the retained server.
