# Ketentuan Pemodelan, Pelabelan Data, dan Peraturan Data Analysis Competition IFEST Unpad 2026

Menanggapi beberapa pertanyaan mengenai metode pemodelan dan pelabelan data, berikut ketentuan yang perlu diperhatikan:

## Penegasan Ketentuan

### 1. Apakah Hyperparameter Optimization (HPO) diperbolehkan?

Ya. Peserta diperbolehkan melakukan **Hyperparameter Optimization (HPO)** menggunakan Optuna maupun fitur HPO yang tersedia dalam framework AutoML.

Namun, penggunaan AutoML **hanya diperbolehkan pada komponen optimasi hyperparameter**. Fitur AutoML lainnya tidak diperbolehkan.

### 2. Apakah penggunaan data eksternal diperbolehkan?

Ya. Peserta diperbolehkan menggunakan data eksternal sebagai tambahan data latih. Namun, seluruh model tetap harus dilatih dari awal. Penggunaan **pretrained model, pretrained weights, maupun proses pretraining** tidak diperbolehkan.

### 3. Apakah pelabelan data diperbolehkan?

Ya. Pelabelan atau perbaikan label hanya diperbolehkan pada data latih, yaitu data train resmi maupun data eksternal yang digunakan sebagai tambahan data latih.

Sebagai contoh, peserta diperbolehkan melakukan pseudo-labeling pada data eksternal yang belum memiliki label, kemudian menggunakannya sebagai tambahan data latih. Namun, pseudo-labeling pada data test resmi tidak diperbolehkan. Peserta tidak boleh menggunakan prediksi model sebagai label sementara pada data test, lalu memasukkan kembali data tersebut ke dalam proses training, validasi, pemilihan model, atau fine-tuning.

Prediksi pada data test hanya diperbolehkan untuk menghasilkan file submission dan tidak boleh digunakan kembali dalam pengembangan model.

### 4. Penggunaan pretrained model, checkpoint, bobot, atau embedding

Penggunaan model, checkpoint, bobot, atau embedding pretrained, seperti IndoBERT, BERT, XLM-R, dan FastText pretrained, tidak diperbolehkan. Arsitektur tersebut tetap boleh digunakan apabila seluruh bobot diinisialisasi secara acak dan dilatih sepenuhnya dari awal.

### 5. Penggunaan lebih dari satu model, ensemble, dan AutoML

Peserta diperbolehkan menggunakan lebih dari satu model atau metode ensemble. Penggunaan AutoML tidak diperbolehkan, kecuali komponen yang hanya digunakan untuk Hyperparameter Optimization (HPO), seperti Optuna.

### 6. Pelabelan, perbaikan label, augmentasi, dan penggunaan data test

Pelabelan, perbaikan label, dan augmentasi hanya diperbolehkan pada data latih. Data test hanya boleh digunakan pada tahap inferensi untuk menghasilkan submission dan tidak boleh dilabeli secara manual, dijadikan pseudo-label, maupun digunakan untuk training, fitting, validasi, atau tuning. Upaya memperoleh label asli data test melalui kebocoran atau sumber yang tidak diperbolehkan termasuk rekonstruksi label dan dilarang.

Penyesuaian ini merupakan **penegasan terhadap aturan yang telah berlaku**. Mohon seluruh peserta membaca kembali bagian **Rules** pada halaman Kaggle dan memastikan metode yang digunakan telah sesuai.

Terima kasih atas perhatian dan kerja samanya. Semangat melanjutkan pengerjaan! 🚀

---

# Peraturan Data Analysis Competition IFEST Unpad 2026

Setiap tim peserta **Data Analysis Competition IFEST Unpad 2026** wajib mematuhi peraturan berikut:

1. Nama tim di Kaggle wajib sesuai dengan nama tim yang didaftarkan saat registrasi. Seluruh anggota harus bergabung dalam satu tim Kaggle melalui fitur **Merge Team** sebelum melakukan submisi.
2. Setiap peserta hanya diperbolehkan menggunakan satu akun Kaggle dan terdaftar dalam satu tim DAC.
3. Peserta diperbolehkan menggunakan data eksternal yang bersifat publik dan gratis. Sumber dan tautan Google Drive data wajib dicantumkan pada bagian teratas notebook.
4. Penggunaan model, checkpoint, bobot, atau embedding pretrained, seperti IndoBERT, BERT, XLM-R, dan FastText pretrained, tidak diperbolehkan. Arsitektur tersebut tetap boleh digunakan apabila seluruh bobot diinisialisasi secara acak dan dilatih sepenuhnya dari awal.
5. Penggunaan VLM, LLM, *generative AI*, dan API model untuk menghasilkan prediksi, label, embedding, atau fitur tidak diperbolehkan.
6. TF-IDF, CountVectorizer, BM25, serta embedding yang dilatih dari awal menggunakan data kompetisi diperbolehkan.
7. Peserta diperbolehkan menggunakan lebih dari satu model atau metode ensemble. Penggunaan AutoML tidak diperbolehkan, kecuali komponen yang hanya digunakan untuk Hyperparameter Optimization (HPO), seperti Optuna.
8. Pelabelan, perbaikan label, dan augmentasi hanya diperbolehkan pada data latih. Data test hanya boleh digunakan pada tahap inferensi untuk menghasilkan submission dan tidak boleh dilabeli secara manual, dijadikan pseudo-label, maupun digunakan untuk training, fitting, validasi, atau tuning. Upaya memperoleh label asli data test melalui kebocoran atau sumber yang tidak diperbolehkan termasuk rekonstruksi label dan dilarang.
9. Setiap tim diberikan kesempatan maksimal **3 submisi per hari** dan dapat memilih maksimal **2 submisi final** untuk penilaian *private leaderboard*.
10. Kerja sama antartim, plagiarisme, pembagian dataset, kode, model, maupun hasil prediksi dilarang dalam bentuk apa pun.
11. Seluruh proses pemodelan wajib dapat direproduksi melalui notebook atau kode yang dikumpulkan.
12. Pelanggaran terhadap peraturan dapat mengakibatkan diskualifikasi. Panitia berhak menetapkan aturan tambahan apabila diperlukan.
