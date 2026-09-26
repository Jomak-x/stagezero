# StageZero — GLB karakter desteği ve geçiş planı

26 Eylül 2026 · Kullanıcının onayıyla uygulama başlatıldı; aşağıdaki güncel uygulama notu ilk incelemeden sonra doğrulanan farkları kaydeder.

## Güncel uygulama notu

İzole `codex/glb-characters` worktree’si koordinatörün doğruladığı `c3deb241f4db76ad8d0479ac91c2b47da80595de` tabanını kullanır. Bu tabanda `director_viewer.py` ve sürümlenmiş `studio_client/` zaten vardır. Bu nedenle eski checkout için aşağıda değerlendirilen bağımlılık fork’u yerine GLB mesajları ve renderer mevcut Studio istemcisine eklenir; kurulu site-packages değiştirilmez. `live_viewer.py` eski yedek olarak kalır. Pod, Ollama, diğer worker checkout’ları ve canlı servisler bu çalışmanın kapsamı dışındadır.

G1 nötr pozu ile insan T/A bind pozu aynı değildir. Sadece nötr dönüş farkını taşımak insanın kollarını T pozunda bırakır. Uygulamada kaynak eklem dönüşüne ek olarak uzuv yönü hizalaması kullanılır; GLB’nin bozulmamış bind pozu ayrı bir `bind_pose()` doğrulamasıyla test edilir. G1 nötr girdisinin hedefin T bind pozunu üretmesi artık kabul ölçütü değildir; beklenen, kaynak uzuv yönlerini hedef uzunluklarıyla takip etmesidir.

Yeni `asset_viewer.py` token/CSV/Pod gerektirmeden yerel GLB ve açıkça etiketlenmiş sentetik eklem hareketiyle doğrulama sağlar. Gerçek inference ve diğer ajanların canlı uygulamasına dağıtım ayrı, koordineli doğrulama aşamasıdır.

İstekteki `.gbl`, devamındaki kullanım doğrultusunda `.glb` olarak yorumlandı. Kullanıcının netleştirdiği hedef: **Patron’a özel bir entegrasyon değil, uygun yapıya sahip farklı GLB modellerinin karakter olarak kullanılabilmesi.** İlk mimari öneri mevcut ARDY G1 hareket üretimini korumak; GLB içe aktarma, uyumluluk kontrolü ve rig profilleri üzerinden genel karakter desteği eklemek. İnsan hareket kalitesi yeterli olmazsa ARDY Core ayrı bir geçiş kararıdır.

“Herhangi uygun GLB” şu anlama gelir: dosya adı/karakter görünümü uygulamaya sabitlenmez; geçerli skin ve kemik yapısı bulunan, desteklenen bir rig profiliyle eşleşen veya açık bir eşleme paketi sağlanan model hareket ettirilebilir. Rastgele her iskeletin otomatik ve doğru retarget edileceği sözü verilmez. İlk hareket ailesi iki ayaklı insansı modellerdir; dört ayaklı, yüz-only veya mekanik rigler için farklı hareket/eşleme gerekir.

## 1. Mevcut sistem ve doğrulanmış bulgular

| Katman | Mevcut davranış | GLB açısından sonuç |
| --- | --- | --- |
| UI | `live_viewer.py:17-62`: Python ile oluşturulan Viser sahnesi ve sağ kontrol paneli. Koyu tema, zemin/platform, hareket kaynağı, komut, Generate, Play, Pause, Reset, kamera takibi. | Ayrı bir uygulama frontend’i yok; ilk sürüm mevcut UI içinde geliştirilebilir. |
| Karakter | `live_viewer.py:29-33`: upstream `Character`, sabit `mesh_mode="g1_stl"`; robot parçalarına tek renk atanıyor. | Dosya yolunu GLB ile değiştirmek yeterli değil; renderer seçimi ayrılmalı. |
| Oynatma | `live_viewer.py:113-129`: `MotionSession` karesindeki global eklem konumları/dönüşleri karaktere uygulanıyor. | Yeni renderer aynı hareket akışını tüketebilir. |
| Oturum | `live_motion.py:64-217`: en yeni komut kazanır; eski yanıtlar sürüm kontrolüyle elenir. Pause, reset ve hareket geçmişi yerelde yönetilir. | Görsel karakter seçimi bu hareket durumundan ayrı tutulmalı. |
| GPU backend | `pod_backend.py:39-45,79-115`: G1, 34 eklem, 25 FPS, iki horizon/104 kare; NPZ yanıt. | G1 korunursa checkpoint ve inference API değişmez. |
| Veri doğrulama | `live_motion.py:52-61`: konum `(104,34,3)`, global dönüş `(104,34,3,3)`, history verisi `(104,414)`. | Retargeting bu veriyi görünüm için dönüştürür; backend’e gönderilen history kaynak biçiminde kalır. |
| API | `pod_backend.py:143-176`: bearer token ile `/health`, `/generate`, `/cancel`; loopback dinleme. | GLB için Pod’a upload endpoint’i eklemek gerekmiyor. |
| Dağıtım | Viewer Mac tarafında, inference Pod’da. Tarayıcıya sahne Viser üzerinden iletiliyor. | GLB ve katalog viewer’ın çalıştığı makinede bulunmalı; Pod’a kopyalanması gerekmez. |

UI değerlendirmesi kaynak kodu ve `review/stagezero-live-wave.png` tarihsel ekran görüntüsüne dayanır. Bu incelemede canlı tarayıcı, mevcut Pod veya yeni GLB animasyonu çalıştırılmadı.

### Mevcut karakter varlıkları

GLB JSON manifestleri incelendi: dört modelde de `skins=0`, `animations=0`. Bunlar hareketli karakter teslimleri değil, statik geometri kaynaklarıdır.

| Varlık | Mevcut içerik | Kullanım önerisi |
| --- | --- | --- |
| `assets/characters/patron/patron-v1.glb` | Ham TripoSR, vertex renkleri; kaynak Z-up. | Arşiv/kıyas; otomatik Y-up varsayımı yapılmamalı. |
| `assets/characters/patron/patron-textured-v1.glb` | 36.264 üçgen, gömülü base-color; yaklaşık 1,74 MB. | Hafif, dokulu GLB yükleme denemesi. |
| `assets/characters/patron/patron-textured-v2.glb` | 147.214 üçgen; yaklaşık 4,09 MB. | Daha ayrıntılı dokulu görünüm kıyası. |
| `assets/characters/patron-blender/patron.glb` | 143 mesh, 15 materyal, 691.290 üçgen, yaklaşık 15,5 MB; doku yerine materyal/vertex renkleri. | Ağır ve çok parçalı statik GLB yükleme sınırını inceleme; hedef karakter değildir. |

Üçgen ve boyut bilgileri mevcut üretim/validation kayıtlarıyla desteklenir. Blender modelinde specular/sheen uzantıları da var; tarayıcıda materyal eşleşmesi ayrıca görülmeli. Çok parçalı geometriyi sadece kemiklere bağlamak omuz, dirsek ve kıyafetlerde düzgün deformasyon sağlamaz.

`assets/characters/` ve önceki `docs/ARDY-ASSET-SPEC.md` şu anda Git tarafından takip edilmiyor. Mevcut dosyalar korunmalı; yeni çalışma çıktıları ayrı adlarla üretilmeli. Ekip kurulumu için runtime varlıklarının sürümlenmesi/dağıtılması ayrıca tamamlanmalı.

## 2. Ana teknik karar: GLB görüntülemek ile GLB’yi oynatmak

Sabitlenmiş `kimodo-viser` bağımlılığında `scene.add_glb` zaten bulunuyor. Dosyanın baytlarını aktararak sahne hiyerarşisi ve materyalleriyle statik görüntüleme yapılabilir. Genel GLB yüklemede trimesh üzerinden flatten/re-export yapmayı varsayılan seçmeyelim; bu dönüşüm materyal ve rig bilgisinin kaybına yol açabilir.

Ancak `GlbHandle` tüm nesnenin dönüşümünü kontrol ediyor; GLB içindeki kemikleri Python’dan sürmeye hazır bir API sunmuyor. Viser’ın ayrı `add_mesh_skinned` yolu kemik güncelleyebiliyor fakat UV/doku materyal parametreleri sunmuyor ve vertex başına dört etki kullanıyor. Dolayısıyla bu iki API’nin birleşik, hazır bir dokulu GLB animasyon çözümü olduğu varsayılmamalı.

Ayrıca mevcut tarayıcı yükleyicisi gömülü animasyon kliplerini otomatik oynatıyor. ARDY modunda bu davranış kapatılmalı veya ilk paketlerde animasyon klibi bulunmaması doğrulanmalı; aksi halde Pause/Reset ile farklı saatlerde çalışan iki animasyon sistemi oluşur.

| Seçenek | Kazanç | Sınır / karar |
| --- | --- | --- |
| Viser `add_glb`, statik | En kısa sürede doğru GLB görünümü ve katalog. | İlk aşama. İskeletsiz modeli yalnız kökten taşımak, karakter animasyonu kabul edilmez. |
| Viser `add_mesh_skinned` | Basit rig ve hareket aktarımını hızla deneyebiliriz. | Tek renkli teknik deneme için uygun; nihai materyalli görünüm hedefini tek başına karşılamaz. |
| Viser GLB kemik kontrolü uzantısı | UI ve oturum korunur; tarayıcı GPU skinning, GLB materyalleri birlikte kullanılır. | Önerilen hareketli yol; önce küçük teknik denemede doğrulanmalı. Python mesajı, istemci işleyicisi, kemik bağlama, hazır/hata bildirimi ve dependency build’i gerekir. |
| Ayrı Three.js frontend | GLB üzerinde tam kontrol. | Yeni frontend, bağlantı, oturum senkronizasyonu ve dağıtım işi getirir. Viser uzantısı başarısız olursa veya kapsamlı UI dönüşümü istenirse değerlendirilir. |

Viser uzantısı gerekiyorsa `.venv/site-packages` içine kalıcı yama yapılmamalı. Kaynağı ve build süreci kayıtlı bir fork/bağımlılık commit’i kullanılmalı; `requirements-preview.txt` içindeki pin yeniden üretilebilir şekilde güncellenmeli. `vendor/ardy` alt modülüne yalnız gerekli olduğu kanıtlanan değişiklikler taşınmalı.

## 3. Önerilen sorumluluk ayrımı

```text
Tarayıcı / Viser sahnesi
    ↑ GLB ilk yükleme + karakter sürümü + kemik pozları
ActorRenderer ← Retargeter ← MotionSession
    ↑                           ↑
CharacterCatalog          Backend istemcisi
    ↑                           ↑
Yerel GLB + manifest      Pod /generate → G1 hareketi
```

Önerilen yeni modüller; mevcut dosyalar değildir:

| Dosya | Sorumluluk |
| --- | --- |
| `character_assets.py` | Katalog, manifest, GLB ön kontrolü, uyumluluk durumu ve varlık özeti. |
| `character_renderer.py` | Ortak renderer arayüzü; mevcut G1, statik GLB ve skinned GLB uygulamaları. |
| `retargeting.py` | Kaynak global pozları hedef rig dönüşlerine çevirme; UI ve ağdan bağımsız hesaplama. |
| `asset_viewer.py` | CSV, token ve Pod gerektirmeyen GLB görünüm/rig deneme girişi. |
| `assets/characters/catalog.json` | Kullanıcıya gösterilen karakterler ve manifest konumları. |
| `rig_profiles/` | Karakterden bağımsız kemik rolleri, kaynak-hedef bind kalibrasyonu ve profile ait uyumluluk kuralları. |
| `.runtime/characters/<asset-id>/` | Kullanıcının içe aktardığı GLB, doğrulanmış manifest ve modele özel eşleme/kalibrasyon bilgisi. |
| `tests/test_character_assets.py`, `tests/test_retargeting.py` | Dosya/rig sözleşmeleri ve dönüşüm regresyonları. |

`live_viewer.py` renderer seçimini ve UI bağlarını yönetir. `MotionSession` hareket üretimi ve oynatma sahibi olarak kalır. Aynı G1 kaynak sözleşmesinde görünüm değiştirmek için `pod_backend.py` veya NPZ veri şekilleri değiştirilmez.

Manifest için asgari alanlar: şema sürümü, sabit kimlik, görünen ad, GLB yolu/hash’i, `static`/`skinned` türü, birim, up/forward yönleri, onaylanmış ölçek, hedef rig profili, desteklenen hareket kaynağı, mapping dosyası, boyut/üçgen özeti ve kaynak/lisans notu. Görüntüleyebilir olmak ile ARDY animasyonuna uyumlu olmak ayrı özelliklerdir. Birden fazla skin/mesh aynı rig profilini kullanabiliyorsa desteklenir; bağımsız ve eşleşmeyen armature içeren modeller açıklayıcı uyumluluk hatası verir.

### Uygun GLB sözleşmesi

- GLB 2.0; geçerli üçgen mesh, normaller ve desteklenen materyaller; kaynaklar dosya içinde olmalı.
- Animasyon için skin, geçerli joint referansları, `JOINTS_0`/`WEIGHTS_0` ve bind dönüşleri gerekir. Inverse bind alanı yoksa glTF varsayılanı uygulanır; bu varsayılanın hedef rig için doğru olduğu ayrıca doğrulanır.
- Kemik rolleri en az pelvis/kök, gövde, iki üst-alt kol ve iki üst-alt bacak/ayak zincirini kapsar. Parmak, twist ve yardımcı kemikler profile göre opsiyoneldir; hareket üretilmeyen kemikler için açık fallback vardır.
- Metre/Y-up iç standardına dönüştürülebilen ölçek ve eksen bilgisi; belirli ve doğrulanmış bind poz. T-pose/A-pose tek başına uyumluluk kanıtı değildir.
- Gömülü animasyon klibi zorunlu değildir; ARDY oynatımında varsa otomatik çalışmaz.
- İskeletsiz dosyalar statik önizlemeye kabul edilir; eksik kemik eşlemesi olan rigli dosyalar “Mapping required” durumunda kalır.

Rig profili, “bu modelin adı ne?” yerine “bu kemik hangi vücut rolünü taşıyor?” sorusunu çözer. İlk teknik denemede G1’e doğrudan uyumlu rig ile yaygın bir insansı rig örneği (örneğin doğrulanmış Mixamo export’u) kullanılır. Tanınan profile otomatik eşleme önerilir; hiyerarşi ve bind kontrolü geçmeden etkinleştirilmez. Bilinmeyen uyumlu rig için manifest/mapping dosyasıyla roller tanımlanabilir. Tam görsel kemik eşleme editörü sonraki iyileştirmedir; ilk sürümde modele özel kod yazılması gerekmez.

## 4. UI davranışı

Mevcut koyu sahne ve kontrol dili korunur. Panelin üstüne **Load GLB**, **Character** seçimi, **Frame character** eylemi ve kısa durum eklenir: “Loading”, “Static preview”, “Mapping required”, “Ready for motion”, “Load failed”. Teknik kemik ve poligon ayrıntıları varsayılan akış yerine isteğe bağlı tanılama alanında bulunur. İçe aktarma sonucu eksik gereksinimler anlaşılır biçimde gösterilir; “uzantı doğru” diye animasyon desteği ilan edilmez.

1. Kullanıcı GLB dosyası yükler veya önceden doğrulanmış katalogdan karakter seçer. Tanınan rig profili ve uyumluluk sonucu gösterilir. Motion source seçimi ayrı kalır.
2. Yeni varlık arka planda hazırlanırken mevcut karakter görünür kalır.
3. Hazır olduğunda en güncel hareket karesi yeni karaktere uygulanır, görünürlük birlikte değiştirilir; eski GPU kaynakları bırakılır.
4. Hata halinde mevcut karakter ve oynatma korunur; hata ve tekrar deneme sunulur. İlk yüklemede G1 yedeği kullanılır.
5. Hızlı A→B→C seçiminde yalnız son seçim etkinleşir. Bunun için inference request sürümünden ayrı `actor_revision` kullanılır.
6. Aynı hareket kaynağına uyumlu karakter değişimi prompt’u, history’yi ve Play/Pause durumunu sıfırlamaz. Duraklatılmış kare de yeni karaktere uygulanır; yalnız kare numarası değiştiğinde render eden mevcut koşul karakter sürümünü de hesaba katmalıdır.
7. Riglenmemiş GLB açıkça “Static preview” gösterir. Bu görünümde Generate/Play desteklenmiyorsa devre dışıdır; çalışan gizli bir hareket oturumu izlenimi verilmez. Aktif oturumdan statik moda başarılı geçiş playback’i duraklatır ve bekleyen isteği geçersiz kılar. Yalnız `pause()` çağrısı iptal sağlamaz; `MotionSession` içinde kilit altında mevcut `_invalidate` davranışını kullanan açık bir geçiş metodu ve geç yanıtın reddi gerekir. Başarısız model yüklemesi bu geçişi tetiklemez.
8. Reset seçili karakteri korur; mevcut anlamıyla hareketi ve kamerayı sıfırlar. “Frame character” yalnız kamerayı değiştirir.

Kamera hedefindeki sabit `0.75` ve mesafeler model bounding box’ına göre hesaplanmalı. Dünya ekseni Y-up, zemin `y=0`; ölçek/eksen dönüşümü tek yerde uygulanmalı. GLB taban noktası ile iskelet pelvis kökü farklıdır; bind kökü, ayak tabanı ve root travel ayrı hesaplanmalı. Kamera takibi de renderer’ın dönüştürülmüş root hareketini kullanmalı.

Mevcut uygulama sekmeler arasında ortak aktör/oynatma durumu paylaşır. İlk sürümde karakter seçimi de ortak olmalı; her sekmeye ayrı avatar oturumu açılması kapsamı büyütür. Sonradan bağlanan sekme güncel GLB, karakter sürümü ve pozu almalı.

Statik aşamadan itibaren istemci yükleme bildirimi `client_id`, `asset_id`, `actor_revision` ve `loaded/error` durumunu taşımalı. Seçimi başlatan istemcinin güncel başarı bildirimi seçim işlemini tamamlar; diğer sekmeler kendi yüklemesi hazır olunca güncel pozu gösterir. Her istemcide eski görünüm yeni görünüm hazır olana kadar korunur. Süre aşımı, başlatan istemcinin işlem tamamlanmadan kopması veya hata işlemden vazgeçilmesine yol açar; eski sürüme ait geç bildirimler yok sayılır. Bekleyen sahne nesneleri ve GPU kaynakları temizlenir. Yeniden bağlanan istemci yalnız güncel seçimi yeniden yükler.

Bu davranış için yüklenen/mevcut görünüm geçişi istemci uzantısında sekme başına yönetilmelidir; ortak `server.scene` görünürlüğünü değiştirmek tek başına yeterli değildir. İlk kez bağlanan sekme yüklenme durumunu gösterir; kendi yüklemesi başarısızsa G1 yedeğine geçer ve ortak seçimin o sekmede gösterilemediğini bildirir. Diğer sekmeler tek bir istemcinin yükleme hatası yüzünden durdurulmaz.

## 5. Rig ve hareket aktarımı

### İlk teknik hedef: G1 kaynağını korumak

Önce basit, düşük poligonlu ve bilinen rigli bir GLB ile aktarımı doğrulayalım. Ardından farklı oranlara ve farklı kemik adlarına sahip ikinci bir GLB ile kodun tek karaktere bağlı olmadığını kanıtlayalım.

- G1’in kalça/omuz/bilek için ayrı yaw/roll/pitch eklemleri vardır. İnsan kemiğine indeks veya ad benzerliğiyle bire bir kopyalama yapılmaz; uygun zincirlerin birleşik dönüşleri ve bind eksenleri eşlenir.
- Backend global dönüş/konum verir. Hedef GLB’nin yerel kemik dönüşlerine geçerken kaynak nötr pozu, hedef bind pozu, ebeveyn hiyerarşisi ve inverse bind matrisleri birlikte ele alınır.
- Hedef dünya dönüşü hesaplandıktan sonra yerel dönüş `local_joint = inverse(world_parent) × world_joint` ilişkisiyle üretilir; skin listesinde bulunmayan ara düğümler de gerçek ebeveyn zincirine dahildir. Hedef bind uzuv uzunlukları/ölçekleri korunur. Root travel hem nesne köküne hem pelvis kemiğine ikinci kez uygulanmaz; testte bilinen 1 metrelik kaynak hareketi kalibrasyon ölçeğiyle beklenen tek yer değiştirmeyi üretmelidir.
- Hedef vücut oranları korunur. Robotun bütün global eklem konumlarını insanın kemik konumlarına kopyalamak uzuvları esnetebilir. Root konumu ve kalibre edilmiş rotasyon aktarımı temel alınır; gerekli ayak düzeltmesi ayrıca uygulanır.
- G1’de bağımsız karşılığı olmayan boyun, parmak veya yardımcı kemikler için sabit/ebeveynden türeyen davranış tanımlanır. Yüz ve dudak animasyonu bu değişimin doğal sonucu değildir.
- Kaynak history `motion` dizisi değiştirilmez. Retarget edilmiş mesh/kemik verisi inference history’si olarak geri gönderilmez.
- Root hareket ölçeği, ayak tabanı yüksekliği ve gerekiyorsa sınırlı IK düzeltmesi ölçülür. Backend şu anda foot-contact alanı göndermiyor; ayak kilitleme hazır veri gibi varsayılmaz.

**Teknik kabul:** nötr poz değişmeden kalmalı; tek eklem denemeleri doğru taraf/eksende dönmeli; bilinen yürüme, squat, kol kaldırma ve dönüş klipleri patlayan mesh, ters diz veya kök kayması üretmemeli.

### Referans model paketi

İki farklı, kullanım hakkı açık ve hazır rigli GLB seçilir. En az biri dokulu/materyalli, diğeri farklı vücut oranları veya kemik adlarıyla eşleme sistemini sınayan örnek olmalı. Model üretmek/otomatik riglemek bu entegrasyonun zorunlu parçası değildir. Test paketinde kaynak, lisans, rig profili ve beklenen nötr görünüm kayıtlı olmalı. Kötü ağırlıklandırılmış modelin deformasyon sorunu ile retargeting hatası ayrıştırılmalı.

Başlangıç performans hedefi olarak yaklaşık 50–100 bin üçgen, azaltılmış mesh/material sayısı ve mümkünse 10 MB altı runtime dosya değerlendirilebilir. Bunlar mevcut ölçülmüş sınırlar değil; hedef makinedeki kalite/performance kıyasına göre güncellenecek bütçelerdir.

### Core’a geçiş kapısı

G1 retarget edilmiş hareket hedef insan karakterlerinde belirgin biçimde robotik kalırsa veya insan eklem kapsamı yetersizse Core tercih edilir. Bu, dosya desteğinden ayrı iş paketidir: Core checkpoint erişimi/kurulumu, `cskel27`, 20 FPS, gerçek checkpoint metadata’sından horizon/özellik boyutu, istemci doğrulaması, başlangıç pozu ve uygun recorded fallback birlikte ele alınır. Farklı iskeletler arasında history taşınmaz. Core da keyfi GLB’yi kendiliğinden uyumlu yapmaz; rig/bind aktarımı ve materyalli renderer yine gerekir.

## 6. Uygulama sırası ve teslimler

| Aşama | İş | Bağımlılık | Somut çıkış / kabul |
| --- | --- | --- | --- |
| 0 — Teknik deneme | Hafif dokulu GLB’yi aç; basit rigli GLB’de bir kemik ve kök sür; materyal, ışık, gölge ve pause davranışını kıyasla. | Mevcut Viser; ayrı asset viewer. | Statik destek kanıtı + materyalli skinning için uygulanabilir yol kararı. |
| 1 — Varlık katmanı | Dosya seçimi/import, katalog/manifest, içerik kontrolü, renderer arayüzü, statik seçim, model kadrajı; sürümlenmiş istemci hazır/hata bildirimi ve timeout/cleanup. | Statik deneme. | İki GLB arasında geçiş; bozuk dosyada mevcut sahne kalır. |
| 2 — Animasyon altyapısı | Viser uzantısında GLB kemik kimlikleri ve poz mesajları; GPU skinning; root/retarget sözleşmesi. | Rigli deneme başarılı. | Basit karakter G1 karesini materyallerini koruyarak izler; Pause/Reset senkron. |
| 3 — Genel rig profilleri | Kemik rolleri, profile göre eşleme, modele özel bind kalibrasyonu, desteklenmeyen rig raporu. | Hedef rig denemesi; referans model seçimi aşama 1 ile paralel yapılabilir. | İki farklı rigli GLB, modele özel kod yazılmadan hareket eder. |
| 4 — Canlı entegrasyon | Seçim yarışları, render anahtarı, kamera takibi, history korunması, reconnect ve fallback. | Aşama 2–3. | Recorded ve Live ARDY’de aynı seçili karakter; geç yanıtlar yanlış aktöre uygulanmaz. |
| 5 — Dağıtım ve QA | Viewer makinesine varlık/dependency dağıtımı, temiz kurulum, görsel/performance testleri ve dokümanlar. | Entegre sürüm. | Yeniden kurulabilir demo; G1’e geri dönüş doğrulanmış. |

Kaba efor: statik destek/import 1–2 geliştirici günü; materyalli kemik kontrolü/retarget teknik denemesi 1–3 gün; genel rig profilleri 2–4 gün; canlı entegrasyon ve regresyonlar 3–6 gün. Bunlar ilk deneme öncesi planlama aralıklarıdır; model üretimi/rigleme, Core geçişi, tam görsel mapping editörü ve kapsamlı frontend yenilemesi dahil değildir.

## 7. Dosya desteğinin sınırları

İlk sürüm kendi kaynaklarını içeren GLB 2.0 dosyalarını kullanıcıdan alır; kontrol edilen varlıkları oturum/katalog üzerinden seçilebilir hale getirir. Referans modeller repo/dağıtım kataloğunda, kullanıcı dosyaları ignored runtime alanında tutulur. `.gltf`, `.fbx`, `.blend` ve `.gbl` bu destekle otomatik kabul edilmez.

Varlık denetimi: dosya boyutu ve GLB header/chunk sınırları; geçerli JSON ve buffer/accessor sınırları; sonlu dönüşümler; desteklenen gerekli uzantılar; gömülü kaynaklar; skinned karakterde geçerli skin/joint/bind/ağırlık verisi; mapping’de eksik/çift kemik ve rig sürümü. Katalog yolları varlık kökü içinde çözülmeli. GPU’ya açılan geometri/doku boyutu da bütçelenmeli; sıkıştırılmış dosya boyutu tek başına yeterli değildir.

Draco yükleyicisi mevcut Viser’da CDN decoder’ına bağlı. İlk denemede sıkıştırmasız GLB tercih edilir. Draco/KTX2/Meshopt gibi optimizasyonlar, decoder desteği ve çevrimdışı dağıtım doğrulanmadan destekleniyor sayılmaz.

Kullanıcı import’unda doğrulama viewer tarafında, sınırlı boyut ve kontrollü runtime depolama ile yapılır. UI dosya seçimi dosyayı kullanıcının tarayıcısından viewer makinesine taşır; tarayıcıdaki yolun Mac mini’de var olduğu varsayılmaz. Depolamada uygulama üretimi asset kimliği kullanılır; kullanıcı dosya adı doğrudan hedef yol olmaz. Dosya yükleme limiti mümkün olan en erken katmanda uygulanır; Viser upload yetenekleri teknik denemede doğrulanır. GLB, `/generate` isteğine konmaz; bu API hareket komutu/history içindir ve mevcut gövde sınırı 2 MB’dir. Python parse başarısı tarayıcı GPU yüklemesinin tamamlandığını kanıtlamaz; yükleme hatası/başarı bildirimi istemciden alınmalıdır.

## 8. Testler ve tamamlanma ölçütleri

Mevcut testler hareket controller’ını kapsar; GLB görünümü veya inference kalitesini kanıtlamaz. Uygulama sırasında mevcut offline test komutu çalıştırılır, yeni davranışlar için aşağıdaki testler eklenir:

- **Varlık sözleşmesi:** geçerli statik/skinned dosya; eksik skin; bozuk chunk; dış kaynak; eksik/tekrarlı mapping; geçersiz ağırlık; bütçe aşımı. Statik dosya önizlemeye kabul edilirken motion-ready sayılmaz.
- **Dönüşüm:** nötr poz kimliği, ebeveyn/çocuk dönüş zinciri, sağ/sol ayrımı, root translation ve ölçek; G1 global → hedef local → dünya uzayında beklenen sonuç.
- **Oturum:** pause sırasında karakter değişimi, yükleme sırasında A→B→C, reset/generate yarışları, bozuk karakterde fallback, geç inference sonucu, sekme yenileme ve yeniden bağlanma.
- **Görsel kalite:** ön/yan/arka nötr poz; omuz, dirsek, diz, el ve kıyafet bükülmeleri; yürüme/squat/wave/turn; ayak kayması ve zemin kesişmesi; renk, alpha, gölge ve normal doğruluğu.
- **Performans:** aynı makine/tarayıcıda G1 ile karşılaştır; ilk modelin görünmesine kadar süre, kare süresi, Python retarget süresi, ağ trafiği, bellek; 20 karakter değişimi ve 10 dakika oynatmada sürekli kaynak artışı olmaması.
- **Zamanlama:** kaynak hareket Live’da 25 FPS, recorded akışta 60 FPS olarak kalır. Tarayıcı render FPS’i bunlardan ayrı ölçülür. 25 FPS yolunda poz uygulama bütçesi 40 ms’nin altında hedeflenir; bu henüz ölçülmüş sonuç değildir.

Kare başına bütün deforme vertex’leri göndermek yerine model ilk yüklemede, kemik pozları sonrasında iletilmeli. Enterpolasyon eklenirse controller saati ve Pause/Reset ile tutarlı olmalı; ilk doğru sürüm için zorunlu değildir.

**Tamamlanma tanımı:** kullanıcı uygun GLB’yi içe aktarır; desteklenen profille veya eşleme paketiyle materyalli karakter recorded ve gerçek G1 hareketini takip eder; en az iki farklı GLB modele özel kod değişikliği olmadan çalışır; uyumsuz model nedenleriyle açıklanır; karakter değişimi aynı kaynak history’sini bozmaz; hata durumunda kullanılabilir sahne kalır; hedef cihazda kabul edilen kalite/performance ölçümleri kaydedilir; temiz kurulum ve geri dönüş yolu belgelenir. Statik GLB görüntüsü tek başına bu hedefin tamamlandığı anlamına gelmez.

## 9. İlgili ama ayrı tutulacak backend işleri

`docs/QA.md` iki mevcut konuyu belgeliyor: token eksikliğinde launcher’ın açıklamasız çıkması ve runtime prompt metriklerinin takip edilen `review/live-metrics.jsonl` dosyasına yazılması. Yeni varlık önizlemesinin Pod/token bağımsız açılması GLB işine doğrudan yardımcıdır; launcher mesajı ve runtime log konumu küçük ayrı işler olarak ele alınabilir. GLB planı genel API yeniden yazımı, çok kullanıcılı oturum veya inference streaming gerektirmez.

## 10. İnceleme kaynakları

- Yerel UI: `live_viewer.py:16-62,113-159`; eski preview: `preview.py:24-54,57-85`.
- Oturum/veri sözleşmesi: `live_motion.py:42-61,64-217`.
- Backend: `pod_backend.py:34-118,121-176`.
- G1 rijit rig: `vendor/ardy/ardy/viz/g1_rig.py:15-51,200-213`.
- Upstream renderer: `vendor/ardy/ardy/viz/viser_utils.py:727-801,858-908`.
- Kurulu Viser GLB/skinning API: `.venv/lib/python3.11/site-packages/viser/_scene_api.py:637-678,1375-1478`.
- Viser istemcisi: aynı paketin `client/src/mesh/GlbLoaderUtils.tsx` ve `SingleGlbAsset.tsx` dosyaları.
- Varlık kayıtları: iki Patron klasörünün README, generation ve validation JSON dosyaları.
- Önceki değerlendirme: `docs/ARDY-ASSET-SPEC.md`; çalışma/QA sınırları: `README.md`, `docs/QA.md`.
