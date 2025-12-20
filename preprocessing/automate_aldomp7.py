import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    """Memuat data mentah ke DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def perform_preprocessing(df):
    """
    Membersihkan data (Drop Duplicates, Remove Outliers IQR, Drop Missing Values).
    Sesuai logika notebook Eksperimen_aldomp7.ipynb.
    """
    # Hapus Duplikat
    df = df.drop_duplicates()
    
    # Hapus Outlier (IQR)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    df_clean = df[condition]
    
    # Hapus Missing Values
    df_clean = df_clean.dropna()
    
    return df_clean

def auto_preprocess_to_csv(filename='preprocessing/namadataset_preprocessing/breastcancer_preprocessing.csv'):
    """
    Pipeline utama: Load -> Clean -> Save to CSV.
    Mengembalikan nama file yang disimpan.
    """
    print("Memulai Automasi Preprocessing...")
    
    # Load
    df = load_data()
    print(f"Data awal: {df.shape}")
    
    # Preprocess
    df_clean = perform_preprocessing(df)
    print(f"Data bersih: {df_clean.shape}")
    
    # Ambil nama folder dari filename
    folder_path = os.path.dirname(filename)
    
    # Cek: Jika folder_path tidak kosong, baru buat foldernya
    if folder_path: 
        os.makedirs(folder_path, exist_ok=True)
    
    # Save to CSV
    df_clean.to_csv(filename, index=False)
    print(f"Data tersimpan sebagai: {filename}")
    print("Selesai.")
    
    return filename

if __name__ == "__main__":

    auto_preprocess_to_csv()
