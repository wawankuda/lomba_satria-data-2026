# 🏆 Satria Data 2026 — Big Data Challenge (BDC)
### Multimodal Waste Classification with Hybrid CAFormer-S18 & FiLM Fusion

Repository ini memuat kode sumber, eksperimen, dan pipeline analitik tim untuk kompetisi **Satria Data 2026 (Divisi Big Data Challenge - BDC)**. Proyek ini berfokus pada klasifikasi citra sampah ke dalam 3 kategori (*Recyclable*, *Electronic*, dan *Organic*) dengan menggabungkan representasi visual citra dan konteks metadata/fitur tabular (*Multimodal Early Fusion*).

---

## 📌 Gambaran Umum Proyek

- **Tugas (Task)**: Klasifikasi Citra Sampah 3 Kelas (`0_Recyclable`, `1_Electronic`, `2_Organic`).
- **Skala Data**: ~26.527 citra latih (*train*) dan 1.458 citra uji (*test*).
- **Pendekatan Utama**:
  - Ekstraksi dan rekayasa fitur tabular berbasis statistik citra (rasio aspek, entropi, distribusi warna).
  - Model *Deep Learning* multimodal dengan **CAFormer-S18** sebagai *vision backbone*.
  - Mekanisme **Feature-wise Linear Modulation (FiLM)** untuk menyuntikkan informasi tabular ke dalam *feature map* spasial.
  - **Spatial Attention Gate** untuk meredam *background noise* dan memusatkan atensi pada objek sampah.
  - *Post-hoc Error Analysis* dan *Interpretability* mendalam untuk mengukur *transfer learning* dan stabilitas fusi.

---

## 🏗️ Arsitektur Model: `HybridCAFormerModel`

Model dirancang dengan paradigma **Early Multimodal Fusion & Auxiliary Supervised Learning**:

```
[ Input Citra (3, H, W) ]           [ Input Tabular (D_tab) ]
          │                                     │
   CAFormer-S18 Backbone                       MLP
(caformer_s18.sail_in22k_ft_in1k)        (Linear -> ReLU -> Linear)
          │                                     │
  Feature Map (512, H, W)                 γ, β (512 x 2)
          ├─────────────────────────────────────┤
          │                                     │
          ▼                                     ▼
[ Auxiliary Image Head ]              [ FiLM Modulation ]
   (Pool -> Linear)              (X_fused = γ * X_feat + β)
          │                                     │
   Loss Aux (Visual)                            ▼
                                    [ Spatial Attention Gate ]
                                    (AvgPool + MaxPool -> Conv7x7 -> Sigmoid)
                                                │
                                                ▼
                                    [ Global Average Pooling ]
                                                │
                                                ▼
                                    [ Joint Multimodal Head ]
                                    (Norm -> Dropout -> Linear -> 3 Kelas)
```

### Detail Komponen:
1. **Vision Backbone**: `caformer_s18.sail_in22k_ft_in1k` (pretrained di ImageNet-22k dan fine-tuned di ImageNet-1k via `timm`) menghasilkan *feature map* berdimensi 512.
2. **FiLM Module (*Feature-wise Linear Modulation*)**: Mentransformasi fitur tabular menjadi parameter modulasi affine $\gamma$ (*scaling*) dan $eta$ (*shifting*) untuk mengondisikan representasi spasial citra.
3. **Spatial Attention Gate (CBAM)**: Filter berbasis konvolusi $7 \times 7$ pada proyeksi channel mean & max untuk mengisolasi objek sampah dari latar belakang rumit.
4. **Dual-Head Optimization**: Cabang *Auxiliary Image-only* melatih *backbone* visual agar tetap independen, sementara cabang *Joint Multimodal* memprediksi label akhir hasil modulasi.

---

## 📂 Struktur Repositori & Alur Pipeline

Alur eksperimen disusun berurutan melalui 3 notebook terstruktur:

```text
bdc_satria_data/
│
├── bdc-2-eda (4).ipynb             # 01. Exploratory Data Analysis & Feature Engineering
├── bcd-2-train (3).ipynb           # 02. Multimodal Deep Learning Training
├── bdc-2-error-analyst (6).ipynb   # 03. Post-Hoc Error Analysis & Interpretability
├── README.md                       # Dokumentasi resmi proyek
```

### 1. `bdc-2-eda (4).ipynb` — *EDA & Feature Engineering*
- Pengecekan integritas dataset, resolusi citra, dan penanganan file rusak (*corrupted images*).
- Analisis distribusi kelas dan ketimpangan (*class imbalance*).
- Ekstraksi fitur tabular dari domain citra:
  - *Aspect ratio*, resolusi total, dan estimasi *bounding box*.
  - Entropi Shannon untuk mengukur kompleksitas tekstur citra.
  - Statistik momen warna (mean, deviasi standar, skewness pada kanal RGB dan HSV).
- Ekspor hasil rekayasa ke file `engineered_features.csv`.

### 2. `bcd-2-train (3).ipynb` — *Model Training & Validation*
- Penggabungan dataset citra dengan data tabular hasil rekayasa fitur.
- Skema validasi **Stratified K-Fold** untuk mencegah kebocoran data (*data leakage*).
- Penanganan ketimpangan kelas dengan *Class-Weighted Cross Entropy*.
- Penggunaan augmentasi data modern (*Mixup*, *CutMix*, *Random Resized Crop*).
- Optimizer **AdamW** dengan *Cosine Annealing Learning Rate Scheduler*.
- Penyimpanan bobot terbaik (*best checkpoints*) per fold.

### 3. `bdc-2-error-analyst (6).ipynb` — *Error Analysis & Interpretability*
- Evaluasi metrik komprehensif (*Macro F1-Score*, *Precision*, *Recall*, *Confusion Matrix*).
- **Ablation Study**: Evaluasi komparatif antara model *Multimodal Joint*, *Vision-Only*, dan *Tabular-Only* untuk mengidentifikasi kontribusi modalitas dan mendeteksi *negative transfer*.
- Analisis stabilitas parameter FiLM ($\gamma, eta$).
- Visualisasi peta atensi spasial (*Spatial Attention Maps*) untuk memahami fokus jaringan pada citra yang salah terprediksi (*hard false positives/negatives*).

---

## 🛠️ Persyaratan Lingkungan (Environment)

- Python >= 3.9
- PyTorch >= 2.0 (CUDA support disarankan)
- timm >= 0.9.12
- albumentations
- scikit-learn
- pandas, numpy, matplotlib, seaborn, PIL

```bash
pip install torch torchvision timm albumentations scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 Cara Menjalankan

Jalankan notebook secara berurutan:
1. Buka dan jalankan seluruh cell pada `bdc-2-eda (4).ipynb` untuk menghasilkan artefak data tabular.
2. Buka `bcd-2-train (3).ipynb` untuk memulai pelatihan model dan menyimpan bobot `best_model_fold_*.pth`.
3. Buka `bdc-2-error-analyst (6).ipynb` untuk menganalisis performa, confusion matrix, dan interpretasi atensi visual.

---

## 👥 Tim & Pengembang
- **Peserta Tim Satria Data 2026**
