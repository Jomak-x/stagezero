# ARDY için karakter ve sahne üretim şartnamesi

26 Eylül 2026 — Bu depodaki StageZero ve vendored ARDY kodu incelenerek hazırlanmıştır. Aşağıdaki mevcut destek bilgileri kaynak koduna dayanır; yeni bir varlık içe aktarılarak çalışma testi yapılmamıştır. Üretim önerileri ayrıca belirtilmiştir.

**Temel karar:** ARDY hareket üretir. Karakterin geometrisini, kaplamasını ve ortamı görüntüleyici yükler. Dosya uzantısının desteklenmesi, karakter iskeletinin üretilen hareketle uyumlu olduğu anlamına gelmez.

| Kullanım | Kodda mevcut yol | Yeni üretim açısından sonuç |
|---|---|---|
| StageZero canlı karakter | G1, 34 eklem, 25 FPS; eklemlere bağlı ayrı `.STL` parçaları | En kısa entegrasyon yolu mevcut robotun parça yapısını korumaktır. |
| ARDY Core insan karakteri | `cskel27`, 27 eklem; `skin_standard.npz` ile ağırlıklı mesh deformasyonu | İnsan veya organik insansı karakter için daha uygun başlangıç; StageZero'nun modele ve veri boyutlarına bağlı kısımları değişmelidir. |
| StageZero arka planı | Kodla oluşturulan zemin ve platform | Henüz harici sahne dosyası yükleme arayüzü yoktur. |
| ARDY'nin ayrı interaktif demosu | `.ply` / `.obj` sahne yükleyicisi | İlk sahne denemesi için kullanılabilir; StageZero'ya ayrıca taşınmalıdır. |
| `.glb`, `.fbx`, `.blend` | StageZero'da genel karakter içe aktarma hattı yok | Üretim/teslim formatı olabilir; mevcut uygulamaya doğrudan takılabilir diye kabul edilmemelidir. |

## 1. Karakter dosyası neleri içermeli?

Üretim önerisi: düzenlenebilir `.blend` kaynak dosyası ve `.glb` teslimi isteyelim. FBX, üretim aracının gerektirdiği alternatif aktarım formatı olabilir. GLB seçimi ileride kurulacak materyalli/iskeletli yükleme hattı içindir; şu anki yerel karakter yükleyicisinin formatı değildir.

- Tam gövde, üçgenlere dönüştürülebilen mesh; hareket sırasında düzgün bükülecek omuz, dirsek, kalça ve diz topolojisi.
- Bir iskelet/armature; eklem adları, sırası, ebeveyn ilişkileri ve kök eklem açıkça tanımlanmış olmalı.
- Nötr/bind pose ve bind dönüşümleri. Rastgele bir T-pose yeterli değildir: hedef iskeletin eklem eksenleri ve nötr yerleşimiyle eşleşmeli veya bunları dönüştüren retargeting adımı bulunmalı.
- Organik karakterde vertex başına kemik indeksleri ve normalize skinning ağırlıkları; kıyafetler de aynı iskelete ağırlıklandırılmalı. Rijit aksesuarlar ilgili kemiğe bağlanabilir.
- Kaplamalı görünüm hedefleniyorsa UV, materyal atamaları ve gömülü ya da eksiksiz teslim edilen dokular. Base color temel; normal, roughness, metallic ve opacity haritaları tasarıma göre eklenebilir.
- Ölçek ve eksen bilgisi: yeni genel varlık hattında metre ve +Y yukarı standardı. İleri yön, kök/pivot ve karakter boyu teslim notunda yazılmalı; içe aktarırken hedef iskeletle doğrulanmalı.
- Kaynak dosya, dışa aktarım ve kısa bir varlık notu: hedef rig, birim, boyut, doku yolları, üretim kaynağı ve kullanım lisansı.

Hazır animasyon klipleri zorunlu değil; hareketi ARDY üretecek. Yüz mimikleri, dudak senkronizasyonu, ayrıntılı parmak hareketi, saç/kumaş simülasyonu ayrıca tasarlanması gereken sistemlerdir. Sadece mesh üretmek bunları sağlamaz.

**Mevcut G1 robotunu koruyacaksak:**

- Çalışan yol tek bir skinned GLB yerine eklemlere bağlı rijit `.STL` parçalardır.
- Parça dosya adları ve kemik eşleşmeleri `G1_MESH_JOINT_MAP` içindedir. İlgili `g1.xml` dosyası parçaların yerel konum/dönüşlerini sağlar; pivotlar bu yerleşimle uyuşmalıdır.
- G1 yükleyicisi mevcut MuJoCo eksenlerini ARDY eksenlerine dönüştürür. Mevcut STL'leri değiştiren dosyalar aynı kaynak koordinat sistemini korumalıdır; genel +Y-up önerisini bu parçalara doğrudan uygulamak ikinci bir eksen dönüşümüne yol açabilir.
- Meshler tek renkli çizilir. Yeni UV ve dokular mevcut robot çizim yolunda kullanılmaz.
- Tek parça bir insan modelini G1 STL klasörüne bırakmak desteklenen bir çözüm değildir.

**Core tabanlı insan/insansı karaktere geçeceksek:**

Hedef `CoreSkeleton27` olmalı veya başka bir rigden Core'a açık bir hareket aktarımı kurulmalıdır. Sadece kemiklerin adını değiştirmek yeterli değildir; hiyerarşi, nötr eklem yerleri, eksenler ve bind dönüşümleri de önemlidir. İlk karakterde hazır Core iskeletini koruyarak mesh'i ona riglemek entegrasyonu kolaylaştırır. Farklı vücut oranları için hareket aktarımı ve ayak teması ayrıca doğrulanmalıdır.

Mevcut Core çizim yolunun okuduğu `skin_standard.npz` alanları:

| Alan | Boyut | İçerik |
|---|---|---|
| `bind_vertices` | `[V, 3]` | Bind pose mesh noktaları |
| `faces` | `[F, 3]` | Üçgenlerin vertex indeksleri |
| `rig_joint_names` | `[J]` | Hedef iskeletle aynı sıradaki eklem adları |
| `bind_rig_transform` | `[J, 4, 4]` | Skinning hesabıyla aynı uzaydaki bind dönüşümleri |
| `lbs_indices` | `[V, W]` | Her vertex'i etkileyen eklemlerin indeksleri |
| `lbs_weights` | `[V, W]` | Aynı etkilerin ağırlıkları |

Burada Core için `J=27`; `V` vertex, `F` üçgen, `W` vertex başına etki sayısıdır. Kaynak yorumunda verilen skin için `W=5` belirtilir; bunu bütün dışa aktarım formatlarının zorunlu sınırı olarak yorumlamamak gerekir. İskeletin nötr eklemleri ayrıca `cskel27/joints.p` dosyasından yüklenir; NPZ tek başına yeni bir iskelet tanımlamaz.

Yükleyici NPZ'yi sabit adla `skeleton.folder` altında arar; mevcut arayüzde bağımsız bir karakter dosyası seçimi yoktur. Birden fazla karakter için varlık seçimi/yolu da eklenmelidir. Dönüştürülmüş verinin kabulünde 27 eklemin adı/sırası ve dönüşüm sayısı, ağırlık indekslerinin aralığı ve vertex başına ağırlık toplamı kontrol edilmelidir.

Core renderer da şu an tek renkli `add_mesh_simple` kullanır; bu NPZ şeması UV/PBR materyal aktarımı sağlamaz. Dokulu karakter için çizim hattı ayrıca genişletilmelidir.

Core'a geçiş sadece karakter dosyasını değiştirmek değildir: backend checkpoint'i, 34 eklem/25 FPS/104 kare/414 özellik varsayımları, başlangıç pozu ve görüntüleyicinin `g1_stl` seçimi birlikte ele alınmalıdır. Depodaki ARDY README'si Core modellerini 20 FPS olarak listeler. SOMA sınıfları bulunmasına rağmen aynı README SOMA checkpoint'ini henüz gelecek sürüm olarak tanımlar; ilk teslimi buna bağlamayalım.

## 2. Arka plan/sahne dosyası neleri içermeli?

İlk uyumluluk denemesi için **tek mesh olarak hazırlanmış `.ply` veya `.obj`** önerilir. Bunlar ayrı ARDY demosunun arayüzünde açıkça belirtilen formatlardır. Yükleyici `trimesh.load` sonucunu doğrudan kullanır ve `.vertices` / `.faces` bekler; çok nesneli bir `Scene` sonucunu birleştiren veya gezen kod içermez. Bu nedenle her çok parçalı OBJ/GLB dosyasının çalışacağı varsayılmamalıdır.

- Statik geometri, üçgen yüzler, doğru normaller ve sahneye uygun gerçek boyutlar.
- Metre ölçeği; +Y yukarı, yatay hareket düzlemi XZ. İlk sahnede yürünecek yüzey `y=0`, karakterin başlangıç çevresi `x=z=0` olacak şekilde hazırlanmalı.
- Blender gibi Z-up kaynaklardan dışa aktarımda dönüşüm uygulanmalı veya upstream demodaki `Z-up to Y-up` seçilmelidir; ikisi birden uygulanmamalıdır.
- Karakterin yürüyüş ve kol hareketleri için açık alan. İlk prototipte düz zemin ve çevresel dekor tercih edilmeli; mevcut platformun çapı yaklaşık 4 metredir.
- Dokulu OBJ tesliminde ilgili `.mtl` ve doku dosyaları da bulunmalı. Gerçek materyal görünümü örnek dosyayla doğrulanmalı; yükleme başarısı görsel doğruluğu tek başına kanıtlamaz.
- Düzenlenebilir `.blend` ve ileride materyal/nesne hiyerarşisini koruyacak yükleyici için `.glb` ana teslimi saklanabilir. GLB içe aktarma desteği ayrıca uygulanıp doğrulanmalıdır.

PNG/JPG tek başına dolaşılabilen 3B ortam değildir. Sabit dekor veya gökyüzü için kullanılabilir; mevcut görüntüleyicide ortam haritası kapalıdır ve ayrı bir resim/skybox yükleme hattı yoktur. `.ply` dosyasının uzantısı da yeterli değildir: yüzleri olmayan nokta bulutu veya Gaussian splat, mevcut üçgen mesh yükleyicisinin beklediği veri değildir.

**Etkileşim sınırı:** Sahneyi çizmek karakterin onu algılamasını sağlamaz. StageZero backend'i sahne geometrisini modele göndermez; mevcut çağrıda kinematik kısıtlar da verilmez. Duvarlardan kaçınma, merdiven çıkma, sandalye üzerine oturma veya nesne tutma için ayrıca sahne bilgisi, hareket kısıtları ve gerekiyorsa çarpışma/temas sistemi gerekir. Upstream demoda kısıt desteği bulunması, yerel uygulamada bu davranışların hazır olduğu anlamına gelmez.

## 3. Üretime verilecek ilk paket

Önerilen ilk deneme: bir insansı karakter ve bir düz zeminli sahne. Önce bu çiftin çalışan aktarımını doğrulayalım; ardından aynı rig ve export ayarlarıyla varyasyonları üretelim.

```text
character_01/
  character.blend          # Düzenlenebilir mesh + armature + ağırlıklar
  character.glb            # Genel teslim; yerel yükleyici henüz desteklemiyor
  textures/                # Harici dokular varsa
  asset-notes.md           # Rig, birim, ileri yön, boy, kaynak/lisans
  skin_standard.npz        # Core yolu seçilirse dönüştürülmüş çalışma verisi

environment_01/
  environment.blend
  environment.glb          # Gelecekteki materyalli yükleme hattı için
  environment.ply          # İlk upstream demo testi için üçgenli tek mesh
  asset-notes.md           # Birim, eksen, zemin yüksekliği, boyutlar
```

PLY yerine OBJ seçilirse gerektiğinde MTL ve dokular eklenmelidir. Bu dizin yapısı üretim önerisidir; uygulamada hazır bir paket keşif sistemi bulunmaz.

Kabul kontrolü: nötr pozda deformasyon olmaması; yürüme, çömelme, kol kaldırma ve dönmede eklemlerin doğru bükülmesi; ayakların zemine oturması; dokuların eksiksiz görünmesi; root hareketinin sahne ölçeğiyle uyuşması; hedef tarayıcıda yükleme ve oynatma performansı. Kodda bir poligon/doku bütçesi tanımlanmadığından sayısal sınırlar örnek varlık ölçülerek belirlenmelidir.

## Kaynaklar

- [Canlı görüntüleyici: sahne ve G1 seçimi](../live_viewer.py#L16)
- [Backend: checkpoint ve sabit iskelet/FPS beklentileri](../pod_backend.py#L20)
- [İstemci: hareket veri boyutlarının doğrulanması](../live_motion.py#L52)
- [İskelet adları, sırası ve hiyerarşileri](../vendor/ardy/ardy/skeleton/definitions.py#L286)
- [Nötr iskelet varlıklarının yüklenmesi](../vendor/ardy/ardy/skeleton/base.py#L70)
- [Core NPZ ve skinning](../vendor/ardy/ardy/viz/core_skin.py#L14)
- [Karakter çizim yolları](../vendor/ardy/ardy/viz/viser_utils.py#L727)
- [G1 parça eşlemesi ve eksen dönüşümü](../vendor/ardy/ardy/viz/g1_rig.py#L15)
- [Upstream sahne yükleme arayüzü](../vendor/ardy/scripts/interactive_demo/gui/io.py#L32)
- [Upstream sahne yükleme uygulaması](../vendor/ardy/scripts/interactive_demo/session_io.py#L578)
- [ARDY model seçenekleri](../vendor/ardy/README.md)
