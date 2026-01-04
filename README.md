# Detektor Mincí (Python Streamlit App)

Jednoduchá webová aplikace pro detekci a počítání českých mincí pomocí Počítačového vidění (OpenCV) a Neuronové sítě (PyTorch).


## Ukázka
![preview](/python_app/galerie/demo1.png)

## Instalace

1. Vytvořte virtuální prostředí (doporučeno):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

2. Nainstalujte závislosti:
   ```bash
   pip install -r requirements.txt
   ```

## Spuštění

Spusťte aplikaci pomocí příkazu:
```bash
streamlit run app.py
```

Aplikace se otevře ve vašem prohlížeči (obvykle `http://localhost:8501`).

## Jak to funguje
1. **Nahrání:** Uživatel nahraje obrázek.
2. **Detekce:** OpenCV (`utils.py`) najde "kolečka" (kandidáty na mince).
3. **Klasifikace:** Neuronová síť (`model.py`) určí hodnotu každé mince.
4. **Výsledek:** Zobrazí se anotovaný obrázek a celková suma.


## Trénování Modelu

Pokud chcete model přetrénovat (např. po stažení datasetu):

1. Ujistěte se, že máte dataset ve složce `../czech-coins` (nebo upravte cestu v `train.py`).
2. Spusťte trénování:
   ```bash
   python train.py
   ```
3. Script automaticky uloží nejlepší model do `coin_model.pth`, který aplikace načte.



Dataset z Kaggle se přes Kaggle API stahuje tak, že nejdřív vygeneruješ API token (`kaggle.json`), umístíš ho na správné místo a pak použiješ příkaz `kaggle datasets download -d ...`.

## 1. Instalace a nastavení API

- Nainstaluj balíček Kaggle API příkazem `pip install kaggle` (Python prostředí / terminál).
- Na stránce účtu na Kaggle v sekci „API“ klikni na „Create New API Token“, stáhne se soubor `kaggle.json` a ten ulož:  
  - Linux/macOS: `~/.kaggle/kaggle.json`  
  - Windows: `C:\Users\<uživatel>\.kaggle\kaggle.json`  
  Soubor má obsahovat uživatelské jméno a klíč, které API používá k autentizaci.

## 2. Stažení datasetu přes CLI

- V terminálu ověř instalaci příkazem `kaggle --help`; pokud proběhla autentizace správně, příkaz proběhne bez chyby.
- Pro dataset (např. `janstol/czech-coins`) použij:  
  - `kaggle datasets download -d janstol/czech-coins` – stáhne ZIP do aktuální složky; můžeš přidat `-p cesta/` pro cílovou složku a `--unzip` pro automatické rozbalení.
