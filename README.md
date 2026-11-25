# 🤖 LLM Hands-On with PyTorch & Hugging Face

Latihan (Task 8) tentang cara **memuat, menjalankan, menyesuaikan, dan menghubungkan Large Language Models (LLM)** dengan **PyTorch** dan pustaka **Hugging Face `transformers`**. Seluruh eksperimen ada dalam satu notebook yang sudah dilengkapi kode, output hasil run, penjelasan tiap bagian, review, dan kesimpulan (dalam Bahasa Indonesia).

> 📓 Notebook: [`notebooks/08_large_language_models.ipynb`](notebooks/08_large_language_models.ipynb)

<!-- Ganti USERNAME dan NAMA-REPO setelah repo dibuat -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/NAMA-REPO/blob/main/notebooks/08_large_language_models.ipynb)

---

## 📚 Isi Notebook

| Part | Topik | Model | Konsep utama |
|---|---|---|---|
| 0 | Environment setup | – | Deteksi `cuda` / `mps` / `cpu` dan pemilihan `dtype` |
| 1 | Autoregressive text completion | GPT-2 (124M) | Loop autoregresif; greedy, temperature, top-k, top-p |
| 2 | Chatbot multi-turn | Qwen2.5-1.5B-Instruct | LLM *stateless*, riwayat chat, *chat template*, loop CLI interaktif |
| 3 | Klasifikasi sentimen | Qwen2.5-1.5B-Instruct | Zero-/few-shot *In-Context Learning* |
| 4 | Ekstraksi informasi (text → JSON) | Qwen2.5-1.5B-Instruct | Menjaga format keluaran agar bisa di-parse |
| 5 | Logit & probabilitas next-token | GPT-2 | `model(**inputs)`, `softmax`, `topk` |
| 6 | Fine-tuning efisien dengan **LoRA** | GPT-2 + `peft` | Melatih ~0,24% parameter, bobot asli dibekukan |
| 7 | **RAG** dari nol | `all-MiniLM-L6-v2` + Qwen2.5 | Embedding, cosine similarity via `torch.matmul`, jawaban *grounded* |

---

## 📊 Ringkasan Hasil

Dijalankan di Google Colab (GPU NVIDIA T4, `float16`), PyTorch 2.14.1, Transformers 5.22.1.

| Eksperimen | Hasil |
|---|---|
| Greedy vs sampling (GPT-2) | Greedy deterministik tetapi berulang; sampling bervariasi tetapi bisa melantur |
| Chatbot multi-turn | Konteks giliran 1 terbawa ke giliran 2 (memori = riwayat pesan yang dikirim ulang) |
| Sentimen few-shot | **3/3** prediksi benar |
| Ekstraksi JSON | Kasus mudah sempurna; kasus sulit ada **halusinasi** (`credits=3` tidak ada di teks) |
| Probabilitas next-token | Identik dengan hasil acuan (`' the'` = 0,0965) |
| LoRA | **294.912** parameter dilatih (**0,2364%** dari 124,7 juta); loss ≈ 4,2 → 0,85 dalam 25 epoch |
| Base vs LoRA | LoRA menjawab "20 GPU hours per week…" (benar, dari data latih); base GPT-2 tidak tahu, tetapi LoRA lalu mengulang dengan angka salah |
| RAG | Skor retrieval 0,773 vs 0,212; jawaban benar; pertanyaan di luar cakupan ditolak ("I do not have that information…") |

Analisis lengkap dan kesimpulan ada di akhir notebook. Karena sampling bersifat acak, beberapa output (Part 1, 2, dan angka loss) **akan sedikit berbeda** di tiap run.

---

## 🚀 Cara Menjalankan

### Opsi A — Google Colab (paling mudah)
1. Klik badge **Open In Colab** di atas (setelah `USERNAME/NAMA-REPO` diganti).
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Jalankan **Runtime → Run all**. Sel pertama akan memasang semua pustaka yang dibutuhkan.

### Opsi B — Lokal
```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# (GPU NVIDIA) pasang PyTorch sesuai versi CUDA Anda: https://pytorch.org/get-started/locally/
pip install -r requirements.txt

jupyter lab notebooks/08_large_language_models.ipynb
```

### Kebutuhan perangkat
| Komponen | Keterangan |
|---|---|
| GPU | Disarankan (≥ 8 GB VRAM). Notebook otomatis memakai `cuda`, `mps` (Apple Silicon), atau `cpu` |
| CPU saja | Bisa berjalan, tetapi Part 2–4 dan 7 (Qwen 1,5B) akan **lambat** dan memakai banyak RAM |
| Disk | ± 4 GB untuk unduhan model (GPT-2 ≈ 0,5 GB, Qwen2.5-1.5B ≈ 3 GB, MiniLM ≈ 90 MB) |
| Internet | Diperlukan saat pertama kali untuk mengunduh model dari Hugging Face Hub |

---

## 🔑 Catatan Hugging Face Token (opsional)

Semua model yang dipakai bersifat **terbuka**, jadi **tidak wajib login**. Peringatan *"unauthenticated requests to the HF Hub"* aman diabaikan. Token hanya berguna agar unduhan lebih cepat atau bila ingin memakai model *gated* seperti `meta-llama/Llama-3.2-1B-Instruct`.

Simpan token di variabel lingkungan atau Colab *Secrets* dengan nama `HF_TOKEN`. **Jangan menaruh token langsung di notebook atau meng-commit-nya.**

---

## 📁 Struktur Repo

```
.
├── notebooks/
│   └── 08_large_language_models.ipynb   # notebook utama (kode + output + dokumentasi)
├── requirements.txt                     # dependensi Python
├── .gitignore
└── README.md
```

---

## ⚠️ Keterbatasan

- Model yang dipakai kecil (GPT-2 124M, Qwen 1,5B), jadi halusinasi dan jawaban ragu-ragu masih terjadi.
- Jumlah contoh uji sangat sedikit (mis. 3 ulasan, 2 pertanyaan RAG); hasilnya bersifat **ilustratif**, bukan evaluasi kuantitatif.
- Dataset LoRA hanya 4 kalimat sehingga model cenderung menghafal (overfitting).
- Perintah `/reset` pada chat interaktif hanya diimplementasikan, belum diuji secara eksplisit di notebook.

---

## 🙏 Kredit

Soal dan kerangka latihan berasal dari materi kuliah **GEN2026 – Task 8: Large Language Models** (© 2026 ADF). Repo ini berisi pengerjaan dan dokumentasinya untuk keperluan belajar.

Model & pustaka: [Hugging Face Transformers](https://github.com/huggingface/transformers) · [PEFT](https://github.com/huggingface/peft) · [Sentence-Transformers](https://www.sbert.net/) · [PyTorch](https://pytorch.org/) · GPT-2 (OpenAI) · Qwen2.5 (Alibaba Cloud) · all-MiniLM-L6-v2 (Sentence-Transformers).
