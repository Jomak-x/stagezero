"""On-demand, viewer-local help for preparing and using a character."""

GUIDE_HTML = '''
<style>
.sz-character-guide { color: #c5d7e3; font-size: 14px; line-height: 1.6; overflow-wrap: anywhere; }
.sz-character-guide p { margin: 0 0 12px; }
.sz-character-guide ol { padding-left: 22px; }
.sz-character-guide li { margin: 8px 0; }
.sz-character-guide b { color: #f0f6fa; }
.sz-character-guide details { border-top: 1px solid #304052; padding: 12px 0; }
.sz-character-guide summary { cursor: pointer; font-weight: 600; color: #a0ead5; }
.sz-character-guide summary:focus-visible { outline: 2px solid #a0ead5; outline-offset: 4px; }
</style>
<div class="sz-character-guide" lang="tr">
<p>Bir GLB dosyasının görünmesi, hareket alabileceği anlamına gelmez.
Hareket için insan biçimli bir iskelet ve modeli kemiklere bağlayan skin ağırlıkları gerekir.</p>
<details open><summary>Hızlı başlangıç</summary>
<ol>
<li><b>Load GLB</b> ile dosyanı seç ve uyumluluk sonucunu bekle.</li>
<li><b>Ready for motion transfer</b> görünüyorsa önce kısa, basit bir hareket dene:
örneğin “wave with the right hand for 3 seconds”.</li>
<li>Oynatıp önden ve yandan kontrol et: omuz, dirsek, kalça ve dizler doğal bükülmeli;
ayaklar ve kaplamalar doğru görünmeli.</li>
<li><b>Frame character</b> ile modeli kadraja al. Sonuç uygunsa daha uzun hareketlere geç
ve ana Studio'da projenin kaydedildiğini kontrol et.</li>
</ol></details>
<details><summary>Dosyayı nasıl hazırlamalıyım?</summary>
<p><b>GLB 2.0</b> kullan; dosya en fazla <b>32 MiB</b> olabilir. Dokuları ve geometriyi dosyanın
içine dahil et. Gereksiz geometriyi ve çok büyük dokuları azaltmak yüklemeyi hızlandırır.</p>
<p>İskelette kalça, omurga, iki kol, dirsekler, iki bacak, dizler ve ayaklar bulunmalı.
Kemikler doğru ebeveyn ilişkilerine ve her hareketli yüzey skin ağırlıklarına sahip olmalı.
Desteklenen Mixamo adları eşleştirmeyi kolaylaştırır.</p>
<p>Modeli doğal ölçekte ve doğru yönde dışa aktar. Mesh ve iskelet dönüşümleri birbiriyle
tutarlı olmalı; negatif veya eksenlere göre farklı ölçekleri rigleme öncesinde düzelt.
Riglenmiş dosyada dönüşümleri körlemesine uygulama; bir kopyada deneyip yeniden kontrol et.</p>
</details>
<details><summary>Uyumluluk sonucu ne anlama geliyor?</summary>
<p><b>Ready for motion transfer:</b> Teknik kontroller geçti. Deformasyon kalitesini yine gözle kontrol et.</p>
<p><b>Bone mapping required:</b> İskelet var, kemik rolleri eksik veya tanınmıyor.
Bu modele ait JSON eşleştirmeyi <b>Load rig mapping</b> ile yükle ve sonucu yeniden kontrol et.</p>
<p><b>Static preview only:</b> Model görüntülenebilir, ancak şu haliyle hareket alamaz.
<b>Open static preview</b> seçersen hareket üretimi kapanır.</p>
<p><b>Unsupported:</b> Dosya açılamıyor veya desteklenmeyen içerik taşıyor.
Gösterilen nedeni incele, Blender'dan uygun GLB olarak yeniden dışa aktar ya da başka dosya seç.</p>
<p>Harekete uygun olmayan bir yükleme, sen sabit önizlemeyi seçene kadar mevcut karakteri değiştirmez.
<b>Rig diagnostics</b> bölümünde ayrıntıları bulabilirsin.</p>
</details>
<details><summary>Modelimde iskelet yoksa ne yapmalıyım?</summary>
<p>Blender'da modelin bir kopyasına insan biçimli iskelet ekle. Eklemleri geometriye yerleştir,
mesh'i iskelete bağla ve otomatik ağırlıklandırmayla başla. Omuz, kalça, dirsek ve dizleri
bükerek ağırlıkları düzelt. Skin, kemikler ve dokularla yeni GLB dışa aktar;
düzenlenebilir .blend dosyasını da sakla.</p>
<p>Bir eşleştirme JSON'u yeni kemik veya skin ağırlığı oluşturmaz. Studio genel otomatik rigleme yapmaz.</p>
</details>
<details><summary>Yere gömülme, kayma veya bozulma varsa</summary>
<p><b>Yere gömülme:</b> Studio yükleme sırasında modeli zemine hizalar; rigli modellerde sabit
ayakta duruşu referans alır. Sorun sürerse Blender'da ayakların altındaki gizli/uzak geometriyi,
skin ağırlıklarını ve model ölçeğini kontrol et.</p>
<p><b>Ayak kayması veya havada kalma:</b> Hizalama her karede tekrarlanmaz; zıplama korunur.
Otomatik ayak kilitleme/IK yoktur. Farklı vücut oranlarında küçük temas farkları kalabilir.</p>
<p><b>Uzayan veya ters dönen uzuvlar:</b> Kemik eşleştirmesini, dinlenme pozunu ve ağırlıkları düzelt.
Yeni hareket üretmek hatalı rig'i onarmaz.</p>
<p><b>Eksik kaplama:</b> Dokuların GLB içine gömülü ve desteklenen biçimde olduğunu kontrol et.</p>
</details>
<p>Karakter seçimi bağlı izleyiciler arasında paylaşılır. Bu rehber yalnızca senin ekranında açılır.</p>
</div>
'''


def add_character_guide(gui):
    button = gui.add_button('GLB kullanım rehberi', color='gray')

    @button.on_click
    def show_guide(event):
        if event.client is None:
            return
        viewer_gui = event.client.gui
        with viewer_gui.add_modal('GLB kullanım rehberi', size='lg', show_close_button=True):
            viewer_gui.add_html(GUIDE_HTML)

    return button
