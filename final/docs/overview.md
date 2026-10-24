# IFEST 2026 Final Data Analysis Competition

Predict next month's water shortages in sub-basins using historical water budget and hydrological data.

## Overview

Ketersediaan air merupakan salah satu faktor penting dalam pengelolaan sumber daya air. Perubahan pasokan dan penggunaan air dari waktu ke waktu dapat menyebabkan terjadinya water stress, terutama pada wilayah dengan kebutuhan air yang tinggi.

Dalam kompetisi ini, peserta ditantang untuk mengembangkan model machine learning yang mampu memberikan peringatan dini terhadap potensi kekurangan air pada bulan berikutnya berdasarkan riwayat kondisi hidrologis dan penggunaan air pada suatu sub-DAS.

Dataset yang digunakan berisi data water budget bulanan pada sub-DAS (HUC12) di wilayah Amerika Serikat selama periode 2000–2020. Data mencakup informasi mengenai pasokan air, penggunaan air, karakteristik wilayah, serta hubungan antarwilayah dalam jaringan sungai.

Peserta diharapkan membangun model yang mampu mengenali pola historis dan memperkirakan tingkat risiko water stress pada bulan berikutnya. Tingkat risiko tersebut dinyatakan sebagai nilai kontinu pada rentang 0 sampai 1, dengan nilai yang lebih tinggi menunjukkan risiko water stress yang lebih besar.

### Description

Kompetisi ini berfokus pada pengembangan model machine learning yang mampu memprediksi potensi water stress pada suatu sub-DAS untuk bulan berikutnya berdasarkan informasi historis mengenai kondisi pasokan dan penggunaan air.

Model yang dikembangkan diharapkan mampu mengenali pola musiman, kondisi hidrologis, kebutuhan air, serta hubungan antarwilayah dalam jaringan sungai untuk menghasilkan prediksi yang akurat.

Melalui kompetisi ini, peserta ditantang untuk membangun model yang tidak hanya memiliki performa prediksi yang baik, tetapi juga mampu melakukan generalisasi pada berbagai sub-DAS dan kondisi hidrologis yang terdapat dalam dataset.

### Evaluation

Submisi akan dievaluasi menggunakan Average Precision (AP), yang juga dikenal sebagai PR-AUC, untuk mengukur kemampuan model dalam membedakan sub-DAS yang berpotensi mengalami water stress pada bulan berikutnya.

Average Precision dihitung berdasarkan hubungan antara Precision dan Recall pada berbagai threshold probabilitas prediksi:

|   |
| - |

```
AP = \sum_n (R_n - R_{n-1})P_n
```

dengan:

Pₙ = Precision pada threshold ke-n

Rₙ = Recall pada threshold ke-n

Precision dan Recall dihitung sebagai:

|   |
| - |

```
Precision = \frac{TP}{TP + FP}
```

|   |
| - |

```
Recall = \frac{TP}{TP + FN}
```

Metrik ini digunakan karena kompetisi merupakan klasifikasi biner dengan kemungkinan distribusi kelas yang tidak seimbang. Peserta harus mengirimkan nilai probabilitas untuk setiap data pada test set, bukan label kelas biner secara langsung.

### Submission Format

Gunakan format berikut untuk file submisi:

```apache
row_id,label
0,0.0142
1,0.7653
2,0.0031

```

File submisi harus memiliki dua kolom:

- `row_id` : ID setiap data pada test set (bilangan bulat), jangan diubah.
- `label` : skor risiko water stress pada bulan berikutnya, berupa nilai kontinu pada rentang 0 sampai 1.

Keterangan skor:

- mendekati 0 : risiko water stress sangat rendah.
- mendekati 1 : risiko water stress sangat tinggi.

Pastikan seluruh row_id pada test set tercantum tepat satu kali dan mengikuti format `sample_submission.csv`.

## Dataset Description

Dataset yang digunakan dalam kompetisi ini terdiri dari beberapa fitur yang digunakan sebagai dasar dalam proses analisis dan pemodelan. Penjelasan mengenai masing-masing fitur, termasuk nama kolom, deskripsi, dan satuan, dapat dilihat secara lengkap pada file **`data_dictionary.csv`** yang disediakan oleh panitia.

## Competition Rules

1. Peserta dilarang menggunakan dataset eksternal dalam bentuk apa pun selama proses pemodelan, analisis, pelatihan, validasi, maupun pengembangan solusi.
2. Penggunaan model pretrained, checkpoint, pretrained weights, maupun pretrained embeddings dalam bentuk apa pun dilarang.
3. Penggunaan arsitektur model kompleks atau standar diperbolehkan, dengan syarat seluruh bobot diinisialisasi secara acak dan model dilatih sepenuhnya dari awal (*from scratch*).
4. Peserta dilarang menggunakan pipeline AutoML secara penuh.
5. Penggunaan tools khusus *Hyperparameter Optimization* (HPO), seperti Optuna, Ray Tune, atau GridSearchCV, diperbolehkan.
6. Setiap notebook wajib menetapkan dan mengunci nilai *random seed* pada bagian awal kode untuk menjamin *reproducibility*.
7. Penguncian *random seed* wajib diterapkan pada seluruh pustaka yang relevan dan digunakan, termasuk namun tidak terbatas pada `random`, NumPy, scikit-learn, XGBoost, LightGBM, PyTorch, dan TensorFlow.
8. File source code notebook wajib dikumpulkan dengan format penamaan: `Final_DAC2026_NamaTim.ipynb`.
9. Notebook yang dikumpulkan wajib dapat dijalankan dari awal hingga akhir tanpa error dan menghasilkan output yang konsisten.
10. Pelanggaran terhadap ketentuan dalam rules ini dapat dikenakan sanksi hingga dan termasuk diskualifikasi berdasarkan hasil pemeriksaan dan keputusan panitia.
