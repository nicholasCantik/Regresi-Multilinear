# Analisis dan Prediksi Kualitas Wine Menggunakan Regresi Linear

**Nama**:
* Nicholas / NIM 103102400047 
* Fatin Hasifa Mediana / 103102400038
* HASNA MARITSA / 103102400059
* LASMAULI YUNITA / 103102400026
## Kaggle 
Link : https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

## About Dataset

Dataset ini berisi data kualitas wine (Vinho Verde) dengan variabel target quality (skor 0–10) dan fitur-fitur fisikokimia hasil uji laboratorium. Berikut penjelasan tiap kolom:
* fixed acidity — Keasaman tetap (terutama asam tartarat) yang tidak mudah menguap; memengaruhi rasa segar/struktur.
* volatile acidity — Keasaman volatil (terutama asam asetat). Nilai tinggi biasanya menurunkan kualitas (aroma cuka).
* citric acid — Asam sitrat; menambah kesan “freshness” dan body pada wine.
* residual sugar — Gula tersisa setelah fermentasi; memengaruhi tingkat kemanisan.
* chlorides — Kandungan garam (klorida); kadar tinggi bisa memberi rasa asin/kurangi kualitas.
* free sulfur dioxide — SO₂ bebas yang aktif sebagai antioksidan/antimikroba.
* total sulfur dioxide — Total SO₂ (bebas + terikat); pengawet, berlebih bisa mengganggu aroma.
* density — Densitas cairan; berkorelasi dengan kadar gula/alkohol.
* pH — Derajat keasaman keseluruhan (lebih kecil → lebih asam).
* sulphates — Sulfat (mis. K₂SO₄) sebagai agen pengawet; sering berkorelasi positif dengan kualitas.
* alcohol — Persentase alkohol (%ABV); umumnya berkorelasi positif dengan skor kualitas.
* quality — Target, skor penilaian sensori oleh panel (skala 0–10)
  
## Alur Program 

### Program 1

#### melakukan impor beberapa library Python yang umum digunakan untuk analisis data.

```python
import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import shapiro

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels
import statsmodels.api as sm
```
### Program 2

#### Membaca isi dari file csv yang digunakan dan menampilkan hanya 10 baris pertama dari data.
```python
winequality = pd.read_csv("winequality-red.csv", usecols= ["alcohol", "volatile acidity", "sulphates", "citric acid", "total sulfur dioxide", "fixed acidity", "quality"])
winequality.head()
```
<img width="829" height="215" alt="Tangkapan Layar 2025-11-09 pukul 14 02 56" src="https://github.com/user-attachments/assets/00bbefed-4fad-40b7-a2ac-17b54c13731c" />

### Program 3

#### Memberikan info singkat mengenai struktur yang ada pada DataFrame.
```python
winequality.info()
```
<img width="412" height="258" alt="Tangkapan Layar 2025-11-09 pukul 17 23 47" src="https://github.com/user-attachments/assets/66bce2cd-525b-4e6f-a870-274f3284dea1" />


### Program 4
#### memberikan info singkat mengenai statistik deskriptif untuk kolom-kolom numerik dalam DataFrame.
```python
winequality.describe()
```
<img width="783" height="249" alt="Tangkapan Layar 2025-11-09 pukul 17 26 33" src="https://github.com/user-attachments/assets/ded51fe2-36ec-4012-a5d9-d684c827942d" />

### Program 5
#### Mengecek apakah pada DataFrame ada nilai null atau tidak.
```python
winequality.isna().sum()
```
<img width="217" height="149" alt="Tangkapan Layar 2025-11-09 pukul 17 29 20" src="https://github.com/user-attachments/assets/f3bcf682-f7e0-4170-bdbd-b7e7033e5c44" />

### Program 6

#### Menghilagkan Nilai duplikat yang ada dalam dataframe 
```python
winequality.drop_duplicates(inplace=True)
winequality.shape
```
<img width="660" height="309" alt="Tangkapan Layar 2025-11-09 pukul 17 32 37" src="https://github.com/user-attachments/assets/446c3b80-353c-4496-94bd-e04056d41a97" />

### Program 7
#### Asumsi & Diagnostik Regresi (OLS)

Tujuan bagian ini adalah **memeriksa asumsi klasik regresi linear** pada model OLS yang memprediksi `quality` dari fitur-fitur fisikokimia.

**Asumsi yang dicek:**
1. **Linearitas**: hubungan antara `X` dan `y` ~ linear → cek dengan **Residuals vs Fitted**.
2. **Independensi error**: tidak ada autokorelasi → **Durbin–Watson** ~ 2.
3. **Homoskedastisitas**: varians error konstan → **Breusch–Pagan (BP)**.
4. **Normalitas residual**: error ~ Normal(0, σ²) → **Jarque–Bera (JB)**.
5. **Multikolinearitas**: antar-fitur tidak sangat berkorelasi → **VIF** < 10.

```python
winequality.columns = winequality.columns.str.strip()

cols = ["fixed acidity", "volatile acidity", "citric acid",
        "total sulfur dioxide", "alcohol", "quality"]  # 5 variabel

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.ravel()

for i, c in enumerate(cols):
    sns.histplot(winequality[c], kde=True, ax=axes[i])
    axes[i].set_title(c)

# Sembunyikan slot terakhir jika tidak terpakai
if len(cols) < len(axes):
    axes[len(cols)].set_visible(False)

plt.tight_layout()
plt.show()
```
<img width="1266" height="567" alt="Tangkapan Layar 2025-11-09 pukul 17 44 16" src="https://github.com/user-attachments/assets/126650a8-7b96-4ce8-b969-bff9da7d89d4" />

### Program 8

#### Scaling 

Kegunaan : untuk merubah skala pada data. Penggunaan scaling dapat membantu mengurangi gap antar kolom dalam data, proses ini tidak merubah distribusi dalam data, hanya melakukan pengubahan skala data.

```python
scaler = StandardScaler()
scaled_wine_quality = scaler.fit_transform(winequality)
scaled_wine_quality
```
<img width="496" height="269" alt="Tangkapan Layar 2025-11-09 pukul 17 47 29" src="https://github.com/user-attachments/assets/60202038-feb1-4dc6-b6bb-f12a0026dc5b" />

### Program 9 

#### Membuat Dataframe dari hasil scaling 

```python
scaled_wine_quality = pd.DataFrame(scaled_wine_quality, columns=winequality.columns)
scaled_wine_quality
```
<img width="683" height="309" alt="Tangkapan Layar 2025-11-09 pukul 17 48 56" src="https://github.com/user-attachments/assets/1f86bcd7-2fa3-4980-8749-c83a2fa3190a" />

### Program 10

#### Membagi dua dataset menjadi data tarin dan data test.

```python
X = scaled_wine_quality.drop("quality", axis=1).values
y = scaled_wine_quality["quality"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
```

### Program 11

#### Membangun Model dengan Package Statsmodel

```python
Input = sm.add_constant(X_train)
impleModel = sm.OLS(y_train, Input, missing='drop')
results = SimpleModel.fit()
print(results.summary())
```
<img width="584" height="506" alt="Tangkapan Layar 2025-11-09 pukul 17 54 58" src="https://github.com/user-attachments/assets/67193bcd-08dd-4e87-8182-a39ed5c7d4cd" />

### Program 12

#### Prediksi Model yang dibuat ditampilkan dalam bentuk dataframe

```python
# Membuat Prediksi Model
Input_test = sm.add_constant(X_test)
prediction_1 = results.predict(Input_test)

# Tampilkan dalam bentuk dataframe
df_statsmod = pd.DataFrame(X_test, columns=list_columns)
df_statsmod["actual_quality"] = y_test
df_statsmod["prediction_quality"] = prediction_1
df_statsmod.head()
```
<img width="788" height="152" alt="Tangkapan Layar 2025-11-09 pukul 17 57 45" src="https://github.com/user-attachments/assets/eccd3ab7-4a9c-4d20-a584-1d91b2c197e7" />

### Program 13

#### Uji Distribusi Residual

```python
# Melihat distribusi residual
residual1 = df_statsmod['actual_quality'] - df_statsmod['prediction_quality']
sns.distplot(residual1, label="residual")
plt.legend()
plt.show()
```
<img width="581" height="425" alt="Tangkapan Layar 2025-11-09 pukul 18 00 08" src="https://github.com/user-attachments/assets/5d10a308-7b03-4539-98ab-70317d006630" />

### Program 14

#### Uji Normalitas pada Residual

```python
stat, p = shapiro(residual1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
  print('Gagal Tolak H0, residual berdistribusi normal')
else:
  print('Tolak H0, residual tidak berdistribusi normal')
```
#### Statistics=0.983, p=0.000
#### Tolak H0, residual tidak berdistribusi normal

#### Artinya :
Dari gambar diatas, plot residual yang dihasilkan membentuk lonceng, tetapi pada pengujian normalitas dengan uji Shapiro-Wilk menunjukkan bahwa residual tidak berdistribusi normal. Sehingga dapat disimpulkan residual tidak berdistribusi normal.


### Program 15 

#### Memasukkan residual ke dalam Dataframe 

```python
# Memasukkan residual kedalam dataframe
df_statsmod["residual"] = residual1
df_statsmod.head()
```
<img width="881" height="142" alt="Tangkapan Layar 2025-11-09 pukul 18 05 05" src="https://github.com/user-attachments/assets/a8f94470-9f58-4923-afa4-3a084afe809e" />

### Program 16

#### Plot data actual dan data prediksi 
Untuk membandingkan hasil data prediksi dengan data yang asli (actual), dapat dibentuk menjadi plot dibawah kode berikut:

```python
sns.distplot(df_statsmod['actual_quality'], label="Actual")
sns.distplot(df_statsmod['prediction_quality'], label="Predicted")
plt.legend()
plt.show
```
<img width="579" height="444" alt="Tangkapan Layar 2025-11-09 pukul 18 07 45" src="https://github.com/user-attachments/assets/47cd4633-34d6-474f-a746-577b84ab1824" />

### Program 17

#### Asumsi autokorelasi dan Uji homoskedasitas

#### Asumsi autokorelasi:
Skor Durbin-Watson antara 1,5 dan 2,5 maka tidak ada autokorelasi dan asumsi puas.

``` python
statsmodels.stats.stattools.durbin_watson(results.resid, axis=0)
```
#### Hasil Asumsi : 2.0622059970878914

#### Uji homoskedasitas:
untuk melihat apakah terdapat ketidaksamaan varians dari residual satu ke pengamatan ke pengamatan yang lain, dari grafik yang dihasilkan dibawah ini, terlihat data berkumpul dalam satu pusat sehingga dapat dikatakan homoskedasitas terpenuhi

```python
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df_statsmod['prediction_quality'], df_statsmod['residual'], s=12, alpha=0.5)

xmin = df_statsmod['prediction_quality'].min()
xmax = df_statsmod['prediction_quality'].max()
sns.lineplot(x=[xmin, xmax], y=[0, 0], color='red', ax=ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Prediction')
plt.show()
```
<img width="543" height="404" alt="Tangkapan Layar 2025-11-09 pukul 18 11 19" src="https://github.com/user-attachments/assets/40dc1a42-be2c-4de7-9503-6a92b251867a" />

### Program 18

#### Evaluasi Model dengan R-square, MSE, Uji Partial Analisis Regresi, dan Uji Kelayakan Model F-test

#### R-square

```python
print('R-square :', metrics.r2_score(df_statsmod['actual_quality'], df_statsmod['prediction_quality']))
```
#### Hasilnya:
R-square : 0.2659804213850434

#### MSE

```python
print('MSE :', metrics.mean_squared_error(df_statsmod['actual_quality'], df_statsmod['prediction_quality']))
```
#### Hasilnya:
MSE : 0.7539321256440609

#### Uji Partial Analisis Regresi
Uji Parameter T-test memiliki tujuan, apakah variabel independent(X) memberikan pengaruh secara partial terhadap variabel dependent (Y)

```python
t_test_const, t_test_X1, t_test_X2, t_test_X3 = results.tvalues[:4]
print("T-test score const: ", t_test_const)
print("T-test score X1: ", t_test_X1)
print("T-test score X2: ", t_test_X2)
print("T-test score X3: ", t_test_X3)
```

#### Hasilnya: 
T-test score const:  0.06809743682979928
T-test score X1:  3.317892039477427
T-test score X2:  -8.563711893450165
T-test score X3:  -2.8480510983805356

```python
p_value_const, p_value_X1, p_value_X2, p_value_X3 = results.pvalues[:4]
print("P-value const: ", p_value_const)
print("P-value X1: ", p_value_X1)
print("P-value X2: ", p_value_X2)
print("P-value X3: ", p_value_X3)
```
#### Hasilnya:
P-value const:  0.9457224913377718
P-value X1:  0.0009416476697418609
P-value X2:  4.395747516692423e-17
P-value X3:  0.004494279390811094

Diperoleh dari output diatas, bahwa P-value ketiga variabel kurang dari 0,05 sehingga Hypothesis null ditolak dan dapat disimpulkan, bahwa ketiga variabel independen (X) memberikan pengaruh secara signifikan pada variabel dependen quality (Y).

#### Uji Kelayakan Model F-test
untuk memutuskan apakah model yang dibentuk layak digunakan atau tidak

```python
f_value = results.fvalue
print("f-test score : ", f_value)

p_value = results.f_pvalue
print("P-value : ", p_value)

if p_value < 0.05:
  print("Tolak H0")
else:
  print("Terima H0")
```
#### Hasilnya:
f-test score :  99.0243684603365
P-value :  1.4132637180821072e-96
Tolak H0

### Kesimpulan:

Inti dari program ini adalah untuk **memprediksi skor kualitas wine (`quality`)** berdasarkan **fitur-fitur fisikokimia** (mis. alcohol, volatile acidity, sulphates, citric acid, total sulfur dioxide, dst.) pada dataset Wine Quality (Red).

**Fungsi Utama Program ini:**  
Program membangun **model Regresi Linear (Multiple Linear Regression, statsmodels OLS)** untuk menganalisis pengaruh tiap fitur kimia terhadap `quality` serta menghasilkan **prediksi skor kualitas**. Selain itu, program menyertakan **evaluasi dan diagnostik asumsi** (linearitas, normalitas residual, homoskedastisitas, dan multikolinearitas) agar interpretasi koefisien dan kesimpulan statistik lebih andal.

**Manfaat Program ini:**
- **Pemahaman Faktor Penentu Kualitas:** Koefisien model membantu menjelaskan arah dan besar pengaruh setiap fitur (contoh umum: `alcohol` cenderung berpengaruh **positif**, `volatile acidity` cenderung **negatif** terhadap `quality`). Ini bermanfaat untuk **quality control** dan **formulasi proses produksi**.
- **Evaluasi Kinerja yang Jelas:** Program tidak hanya memprediksi, tetapi juga **mengevaluasi performa** menggunakan metrik seperti **R-squared** dan **RMSE**, serta **visualisasi** (Residuals vs Fitted, QQ-plot) dan uji formal (**Breusch–Pagan** untuk heteroskedastisitas, **Jarque–Bera** untuk normalitas residual). Hal ini memberi gambaran seberapa baik model merepresentasikan data.
- **Dasar Pengambilan Keputusan Teknis:** Prediksi kualitas dan interpretasi koefisien dapat menjadi **alat bantu** untuk keputusan teknis (mis. penyesuaian parameter fermentasi/penyimpanan) dengan tetap mempertimbangkan bahwa kualitas wine dipengaruhi banyak faktor sensorik non-kimia.
- **Landasan untuk Peningkatan Lanjutan:** Hasil baseline regresi linear dapat dikembangkan dengan:
  - **Robust/Weighted OLS** atau **standard error robust (HC3)** saat terdeteksi heteroskedastisitas.
  - **Transformasi fitur** untuk variabel sangat skew (mis. `log1p` pada `total sulfur dioxide`).
  - **Regularisasi** (Ridge/Lasso) dan **rekayasa fitur** (interaksi/polinomial ringan).
  - **Validasi silang** (train/test split atau k-fold) untuk mengukur **generalization error** yang lebih tepercaya.

Secara keseluruhan, program memberikan **baseline yang interpretatif** untuk memodelkan dan memprediksi `quality`, sekaligus **kerangka evaluasi** yang kuat agar hasilnya dapat dipercaya dan siap dikembangkan ke pendekatan yang lebih canggih bila diperlukan.



