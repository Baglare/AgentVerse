# AgentVerse M1

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
