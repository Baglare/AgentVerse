# AgentVerse — M1 / M2

Python 3.12, Gymnasium, NumPy ve Matplotlib ile sabit haritada tabular Q-learning.
Sonuçlar ve beceri sürümleri Git dışında tutulan `outputs/` altında saklanır.

## Kurulum ve M1 komutları

Proje kökünde PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m agentverse train --agent Alph --episodes 10000 --seed 42
.\.venv\Scripts\python.exe -m agentverse evaluate --agent Alph --episodes 100 --seed 123
.\.venv\Scripts\python.exe -m agentverse baseline --episodes 100 --seed 123
.\.venv\Scripts\python.exe -m agentverse report --agent Alph
```

## Görsel replay

Mevcut kayıtlı Alph için **yeniden eğitim gerekmez**. `trained` modu son kayıtlı
navigation sürümünü bütünlük kontrolleriyle yükler; kayıt yoksa hata verir.
Kararlar Q-tablosunun deterministik `argmax` sonucudur. Her hareket mevcut
`NavigationWorld.step()` üzerinden yürütülür; öğrenme veya kayıt güncellemesi yapılmaz.

```powershell
Set-Location 'C:\AgentVerse\AgentVerse'
$env:MPLBACKEND = 'TkAgg'
.\.venv\Scripts\python.exe -m agentverse replay --agent Alph --seed 123
```

Rastgele ajan (model dosyası gerektirmez):

```powershell
.\.venv\Scripts\python.exe -m agentverse replay --mode random --seed 123 --interval-ms 150
```

- **Adim:** otomatik oynatmayı durdurur ve tek gerçek ortam adımı ilerler.
- **Baslat / Duraklat:** zamanlayıcıyla ilerlemeyi açar veya durdurur.
- **Sifirla:** oynatmayı durdurur; aynı seed ile aynı başlangıcı ve rastgele karar dizisini yeniden kurar.
- Hedef veya 100 adım sınırında oynatma durur. Yeniden izlemek için **Sifirla** kullanılır.

İki modda aynı seed aynı başlangıcı kullanır; bu, aynı seed ile M1 değerlendirmesinin
ilk bölümüdür. Tekrar üretim için aynı model sürümü kullanılmalıdır; yeni eğitim
sonrasında son sürüm değişir. `--output-dir` farklı kayıt kökü seçer.
Haritada koyu kareler duvar, yeşil yıldız hedef, mavi daire ajandır.
Başlıkta seed, adım, toplam ödül, son hareket ve çarpışma gösterilir.

Replay etkileşimli Matplotlib backend gerektirir. Standart Windows Python kurulumundaki
Tk/Tkinter ile `TkAgg` kullanılabilir; `Agg` pencere açmaz ve replay hata verir.
Bu oturum için backend seçimini kaldırmak: `Remove-Item Env:MPLBACKEND`.

### GIF / MP4 değerlendirmesi

Bu sürümde dışa aktarma komutu eklenmedi. Aynı ortam adımlarından alınan kareler
Matplotlib `PillowWriter` ile GIF'e dönüştürülebilir; Pillow, mevcut Matplotlib
kurulumunun bağımlılığıdır. MP4 için Matplotlib `FFMpegWriter` kullanılabilir ancak
ayrıca erişilebilir FFmpeg programı gerekir; mevcut Python bağımlılıkları bunu garanti
etmez. FFmpeg kurulumu veya yeni zorunlu bağımlılık eklenmedi. Hiçbir çıktı biçimi
görülmemiş haritalarda genelleme kanıtı oluşturmaz.

## Testler

Pencere açmadan mevcut M1 ve replay testleri:

```powershell
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -m pytest tests/test_m1.py tests/test_replay.py -q
```

Replay açmadan önce `$env:MPLBACKEND = 'TkAgg'` seçimini geri yapın.

## M2 — Multi-Skill Continuity

Alph aynı `agent_id` altında `navigation` ve `resource_collection` becerilerini
bağımsız Q-tabloları, eğitim geçmişleri, ortam sözleşmeleri ve sürümleriyle tutar.
Bütün komutlarda varsayılan beceri `navigation` olarak kalır. Yeni beceri önceki
Navigation kayıtlarını taşımaz veya değiştirmez. Her eğitim yeni UUID sürümü ekler;
geçmiş sürümler silinmez veya üzerine yazılmaz. Bu sürümde beceriler arasında bilgi
aktarımı yoktur; M3 ve çok ajanlı M4 uygulanmamıştır.

Resource Collection haritası (`B`: üs, `R`: kaynak, `#`: duvar):

```text
B.....
.##.#.
..R.#.
.#....
...#R.
......
```

Başlangıç daima üstür. Harita dışı hücreler de geçilemez. Gözlem,
`toplanan_kaynak_bit_maskesi * 30 + konum_indeksi` olarak kodlanır: 120 durum,
4 hareket, float64 Q-tablosu `(120, 4)`. Her hareket −0.01, çarpışma ek −0.05,
bir kaynağın ilk toplanması +0.2, iki kaynakla üsse dönüş +1 ödül verir.
Tekrar kaynak ziyaretinde toplama ödülü yoktur; ödül üretmeyen döngüler negatiftir.
Yalnız iki kaynakla üsse dönüş başarıdır; diğer bölümler en fazla 100 adım sürer.

PowerShell (mevcut Alph artık bu beceriye de sahiptir; tekrar eğitim yeni sürüm ekler):

```powershell
.\.venv\Scripts\python.exe -m agentverse train --agent Alph --skill resource_collection --episodes 10000 --seed 42
.\.venv\Scripts\python.exe -m agentverse evaluate --agent Alph --skill resource_collection --episodes 100 --seed 123
.\.venv\Scripts\python.exe -m agentverse baseline --skill resource_collection --episodes 100 --seed 123
.\.venv\Scripts\python.exe -m agentverse report --agent Alph --skill resource_collection
$env:MPLBACKEND = 'TkAgg'
.\.venv\Scripts\python.exe -m agentverse replay --agent Alph --skill resource_collection --seed 123
.\.venv\Scripts\python.exe -m agentverse replay --agent Alph --skill navigation --seed 123
```

`evaluate`, `report` ve `replay` için `--version <UUID>` geçmiş bir beceri sürümünü
seçer; verilmezse seçilen becerinin son sürümü yüklenir. Python API karşılığı
`AgentStore(...).load(skill="resource_collection", version="<UUID>")` olur.
Rastgele replay için `--mode random` kullanılır; `--version` kabul edilmez.
Collection replay'de üs yıldız, kaynaklar turuncu elmas, ajan mavi dairedir.
Toplanan elmas kaybolur; başlık kaynak sayısını, adımı ve başarı durumunu gösterir.

Kayıtlar `agents/<isim-hash>/<skill>/<version>/` altında tutulur. Yeni becerinin
raporları/değerlendirmeleri kendi beceri dizinindedir; M1 değerlendirme ve rapor
yolları korunur. Yükleme agent/skill/sürüm eşleşmesini, ortam sözleşmesini,
Q-tablosunun boyutunu/tipini/sonlu değerlerini ve SHA-256 değerini doğrular;
NPZ pickle açmaz. Yeni kayıtlarda eğitim geçmişinin SHA-256 değeri ve bölüm
sayısı da doğrulanır. Eski M1 geçmişlerinde checksum bulunmadığından yalnız
biçim ve bölüm sayısı doğrulanır; kayıtlar yeniden yazılmaz. Bu kontroller
kazara bozulmayı yakalar; imzalı kayıt veya güvenilmeyen dosyalar için sandbox değildir.

### Gerçek yerel deney — 25 Ekim 2026

Collection: 10.000 eğitim bölümü, seed 42; değerlendirme 100 bölüm, seed 123.
Aynı ortam başlangıçlarıyla bağımsız rastgele aksiyon akışı karşılaştırıldı.

| Politika | Başarı | Ortalama adım | Ortalama ödül |
|---|---:|---:|---:|
| Collection Q-learning | %100 | 16.00 | 1.2400 |
| Collection rastgele | %2 | 99.81 | −2.6811 |
| Korunan Navigation Q-learning | %100 | 4.05 | 0.9595 |

Collection sürümü: `615e9a65-6431-462f-bcc3-2b5725a97d06`.
Alph kimliği: `1dd32bf6-05dc-46a2-b7a6-f65aa2271f1e` (değişmedi).
Mevcut iki Navigation sürümü ayrı ayrı yüklendi. M2 eğitimi öncesinde kaydedilen
14 dosyanın (kimlik, Navigation kayıtları ve mevcut rapor/değerlendirmeler)
SHA-256 değerleri sonrasında aynı kaldı. Navigation'ın seed 123 ile 100 bölümlük
bütün değerlendirme kayıtları da birebir aynı kaldı.
Yerel kanıtlar: `outputs/m2-evidence/before.json` ve `result.json`.
Bu dosyalar ve modeller Git'e dahil değildir.

Sabit üs ve sabit harita nedeniyle Collection'ın greedy değerlendirme bölümleri
aynı rotayı tekrarlar. Bu sonuç farklı haritalara genelleme veya beceriler arası
transfer kanıtı değildir. Tekrar üretim testi aynı seed ile 3.000 bölümlük iki
eğitimin Q-tablolarını ve tüm eğitim geçmişlerini birebir karşılaştırır.

M1/replay testleri değişiklik öncesi **24 PASS**; M2 dahil son kontrol **47 PASS**.
Gymnasium sözleşmesi, kaynak ödülü/başarı/zaman sınırı, tekrarlanabilirlik,
kimlik/hash korunumu, eski kayıt yükleme, bozuk dosya reddi, sürüm koruma ve iki
becerinin replay akışı pencere açmadan test edilir.

```powershell
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = "$PWD/outputs/mplconfig"
.\.venv\Scripts\python.exe -m pytest tests/test_m1.py tests/test_replay.py tests/test_m2.py -q --basetemp=outputs/m2-manual-tests
```

Etkileşimli pencereyle manuel görsel kabul **NOT_RUN**. Manuel kontrol için yukarıdaki
iki replay komutunda `TkAgg` kullanın; Adim/Baslat/Duraklat/Sifirla kontrollerini,
kaynakların kaybolmasını ve iki kaynakla üsse dönüşte başarıyı kontrol edin.
M2 için otomatik doğrulama tamamlandı; manuel görsel kabul ayrı tutulur.
Yeni bağımlılık, ücretli API, sunucu veya GPU gerekmez.
