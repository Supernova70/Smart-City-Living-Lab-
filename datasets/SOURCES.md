# Real training datasets

## KADID-10k

- Official page: https://database.mmsp-kn.de/kadid-10k-database.html
- Archive: https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip
- Published size: approximately 3.1 GB (HTTP content length: 3,067,408,471 bytes)
- Contents: 81 pristine photographs, 10,125 distorted images, 25 distortion types, five severity levels, and subjective quality scores in `dmos.csv`.
- Use: non-commercial academic model training and evaluation; cite Lin, Hosu, and Saupe (QoMEX 2019).

Resume-capable download on Windows:

```powershell
curl.exe -L --retry 3 --continue-at - --output datasets\kadid10k.zip https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip
Expand-Archive datasets\kadid10k.zip datasets\kadid10k
```

## MVTec AD

- Official page: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Reproducible mirror: https://huggingface.co/datasets/Voxel51/mvtec-ad
- Contents: more than 5,000 real industrial images in 15 categories, normal training images, anomalous and normal test images, and pixel-precise anomaly masks.
- License: CC BY-NC-SA 4.0; academic/non-commercial use only.

Reproducible mirror download (images, metadata, and license only):

```powershell
python -m pip install -r requirements-train.txt
python -m scripts.download_mvtec --output datasets\mvtec_ad
```

Integrity verification and manifest generation:

```powershell
python -m scripts.verify_real_datasets
python -m scripts.build_real_manifest --kadid-root datasets\kadid10k --mvtec-root datasets\mvtec_ad
```

Downloaded archives and extracted images are excluded from Git. Dataset manifests, split metadata, metrics, and source documentation remain versioned.
