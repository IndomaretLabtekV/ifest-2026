## Overview

Di tengah pesatnya arus informasi digital, judul berita menjadi bagian pertama yang digunakan pembaca untuk memahami suatu peristiwa. Oleh karena itu, judul seharusnya mampu merepresentasikan informasi utama dalam isi berita secara tepat. Ketidaksesuaian antara judul dan isi dapat menimbulkan kesalahpahaman, terlebih ketika keduanya membahas topik atau tokoh yang serupa, tetapi merujuk pada peristiwa yang berbeda.

Mendeteksi ketidaksesuaian tersebut bukan sekadar mencari kesamaan kata. Dua teks dapat memuat banyak kata kunci yang sama, tetapi berbeda dalam penyebutan tokoh, lokasi, waktu, angka, tindakan, maupun konteks peristiwa. Sebaliknya, judul dan isi yang menggunakan susunan kata berbeda belum tentu tidak sesuai apabila tetap menyampaikan informasi yang sama.

Melalui kompetisi ini, peserta ditantang untuk mengembangkan model klasifikasi yang mampu memahami hubungan antara judul dan isi berita serta menentukan apakah keduanya *sesuai* atau *tidak sesuai*.

### Description

## Problem Statement

Peserta diminta membangun model klasifikasi biner berdasarkan pasangan judul dan isi berita. Untuk setiap pasangan pada data uji, model harus memprediksi salah satu dari dua label berikut:

- *Sesuai*, apabila informasi utama dalam judul didukung dan dibahas dalam isi berita.
- *Tidak sesuai*, apabila informasi utama dalam judul tidak didukung oleh isi, membahas peristiwa yang berbeda, atau memiliki perbedaan konteks yang mengubah makna berita.

Model diharapkan tidak hanya mengandalkan kemiripan kata, tetapi juga mampu mengenali hubungan kontekstual antara kedua teks. Hal ini penting karena beberapa pasangan dapat memiliki topik, tokoh, lokasi, atau kata kunci yang serupa meskipun sebenarnya menyampaikan informasi yang berbeda.

Dataset kompetisi terdiri atas pasangan judul dan isi berita. Data training dilengkapi dengan label kesesuaian yang dapat digunakan untuk melatih dan memvalidasi model, sedangkan data uji tidak memiliki label dan harus diprediksi oleh peserta.

## Tujuan

Kompetisi ini bertujuan mendorong peserta untuk mengembangkan model klasifikasi teks yang akurat dan memiliki kemampuan generalisasi yang baik dalam menilai kesesuaian judul dengan isi berita. Peserta diharapkan mampu melakukan eksplorasi data, menerapkan teknik pemrosesan teks yang tepat, membangun representasi fitur yang informatif, serta memilih pendekatan pemodelan yang sesuai.

Model yang dikembangkan diharapkan mampu menangani beragam tingkat kesulitan, mulai dari pasangan dengan topik yang jelas berbeda hingga pasangan yang memiliki kemiripan tinggi.

### Evaluation

Pada kompetisi ini, performa model peserta akan dievaluasi menggunakan **Macro F1-Score** sebagai metrik utama.

Macro F1-Score menghitung F1-Score untuk setiap kelas, yang kemudian dibagi dengan jumlah kelasnya.

Secara matematis:

|   |
| - |

```
F1_{\text{macro}} = \frac{F1_0 + F1_1}{2}
```

Nilai Macro F1-Score berada pada rentang 0 hingga 1. Nilai yang lebih tinggi menunjukkan performa klasifikasi yang lebih baik pada kedua kelas.

### Submission File

Untuk setiap data pada test set, Anda harus memprediksi label kesesuaian judul-isi.

File submission harus berbentuk CSV dengan header, serta memiliki dua kolom berikut.

- `id`, yang berupa identifier pada test set, contoh `te00001`
- `label`, berisi hasil prediksi `1` (Sesuai) atau `0` (Tidak Sesuai)

Contoh submission.csv:

```autohotkey
id,label
te00000,1
te00001,0
te00002,1

```

Pastikan format submission sesuai dengan `sample_submission.csv` yang disediakan.

## Dataset Description

Dataset pada kompetisi ini terdiri dari pasangan judul dan isi berita yang diklasifikasikan ke dalam dua kategori berikut:

1. *Sesuai (1)*: Informasi utama yang disampaikan dalam judul didukung dan dibahas dalam isi berita.
2. *Tidak Sesuai (0)*: Informasi utama dalam judul tidak didukung oleh isi berita, merujuk pada peristiwa yang berbeda, atau memiliki perbedaan konteks yang mengubah makna berita.

Setiap pasangan dapat memiliki variasi dari segi topik, tokoh, lokasi, angka, waktu, dan konteks peristiwa. Beberapa judul dan isi juga dapat mengandung kata-kata yang serupa meskipun sebenarnya membahas informasi yang berbeda.

Dataset terdiri dari beberapa kolom berikut:

- *id*: Identitas unik untuk setiap pasangan judul dan isi berita.
- *title*: Judul berita yang akan dinilai kesesuaiannya dengan isi berita.
- *content*: Isi atau badan berita yang menjadi dasar penilaian kesesuaian.
- *label*: Target klasifikasi yang menunjukkan kesesuaian antara judul dan isi berita, dengan nilai 1 untuk **Sesuai** dan 0 untuk **Tidak Sesuai.**
