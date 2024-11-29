# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(page_title="Data Analysis & ML App", layout="wide", page_icon="📊")

# Sidebar navigation
def show_sidebar():
    st.sidebar.title("Navigasi")
    option = st.sidebar.radio(
        "Pilih Proses:",
        ["📜 Deskripsi Aplikasi", "📂 Data Preparation", "📊 EDA", "📈 Modeling"]
    )
    st.sidebar.info("""
    Jika tidak ada dataset yang diunggah, dataset default **Healthcare-Diabetes.csv** akan digunakan.
    """)
    return option

# Function to load datasets
def load_dataset(upload_key):
    uploaded_file = st.file_uploader(f"Upload Dataset untuk {upload_key} (CSV)", type=["csv"], key=upload_key)
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil dimuat!")
            return data
        except Exception as e:
            st.error(f"Gagal memuat dataset: {e}")
            return None
    else:
        # Load the default dataset
        try:
            data = pd.read_csv("/mnt/data/Healthcare-Diabetes.csv")  # Path ke dataset default
            st.info("Dataset default digunakan: Healthcare-Diabetes.csv")
            return data
        except Exception as e:
            st.error(f"Gagal memuat dataset default: {e}")
            return None

# Function to handle missing values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy="median")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Identify and drop completely empty columns
    empty_columns = [col for col in numerical_cols if data[col].isnull().all()]
    if empty_columns:
        st.warning(f"Kolom kosong sepenuhnya akan dihapus: {empty_columns}")
        data = data.drop(columns=empty_columns)
        numerical_cols = [col for col in numerical_cols if col not in empty_columns]
    
    # Impute missing values
    imputed_data = imputer.fit_transform(data[numerical_cols])
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols, index=data.index)
    data[numerical_cols] = imputed_df
    return data

# Function for exploratory data analysis
def perform_eda(data):
    st.write("### 📊 Statistik Deskriptif")
    st.dataframe(data.describe())

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 0:
        selected_col = st.selectbox("Pilih Kolom untuk Visualisasi Distribusi:", numerical_cols)
        if selected_col:
            st.write(f"### 🔢 Distribusi Kolom: {selected_col}")
            fig = px.histogram(data, x=selected_col, nbins=30, title=f"Distribusi {selected_col}")
            st.plotly_chart(fig)

        st.write("### 🌡️ Korelasi Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig_corr)
    else:
        st.warning("Dataset tidak memiliki kolom numerik untuk analisis.")

# Function for modeling
def perform_modeling(data):
    target = st.selectbox("🎯 Pilih Target Variable:", data.columns)
    features = data.drop(columns=[target])
    X = features.select_dtypes(include=['float64', 'int64'])
    y = data[target]

    if X.empty or y.empty:
        st.error("Dataset tidak memiliki kolom fitur atau target yang valid.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    st.success("Model berhasil dilatih!")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write("### 📊 Evaluasi Model")
    st.metric("🎯 Akurasi", f"{accuracy:.2f}")
    st.text("📋 Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Main app logic
option = show_sidebar()

if option == "📜 Deskripsi Aplikasi":
    st.title("📜 Deskripsi Aplikasi")
    st.write("""
    ### Selamat Datang di Aplikasi Analisis Data & Machine Learning
    Aplikasi ini dirancang untuk membantu pengguna dalam:
    - 📂 **Data Preparation**: Membersihkan dan mempersiapkan dataset.
    - 📊 **Exploratory Data Analysis (EDA)**: Melakukan analisis data eksploratif.
    - 📈 **Modeling**: Melatih model machine learning menggunakan dataset.

    ### Cara Menggunakan
    1. Pilih proses yang ingin dilakukan dari menu navigasi di sidebar.
    2. Upload dataset dalam format CSV jika diperlukan.
    3. Jika tidak ada dataset yang diunggah, dataset default akan digunakan.
    """)

elif option == "📂 Data Preparation":
    st.title("📂 Data Preparation")
    st.write("### 🛠️ Membersihkan Dataset Anda")
    data = load_dataset("Data Preparation")
    if data is not None:
        st.write("### 📋 Dataset Overview")
        st.dataframe(data.head())
        st.write("### ✅ Pembersihan Missing Values")
        data = handle_missing_values(data)
        st.dataframe(data.head())
        if st.button("💾 Simpan Dataset Bersih"):
            data.to_csv("Cleaned_Dataset.csv", index=False)
            st.success("Dataset bersih berhasil disimpan sebagai 'Cleaned_Dataset.csv'.")

elif option == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("### 🔍 Analisis Data Anda")

    # Memuat dataset untuk EDA
    data = load_dataset("EDA")
    if data is not None:
        st.write("### 📋 Dataset Overview")
        st.dataframe(data.head())

        # Statistik Deskriptif
        st.write("### 📊 Statistik Deskriptif")
        st.dataframe(data.describe())

        # Kolom Numerik untuk Analisis
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Visualisasi Distribusi untuk Kolom Numerik
        if len(numerical_cols) > 0:
            selected_num_col = st.selectbox("Pilih Kolom Numerik untuk Visualisasi Distribusi:", numerical_cols)
            if selected_num_col:
                st.write(f"### 🔢 Distribusi Kolom: {selected_num_col}")
                fig = px.histogram(data, x=selected_num_col, nbins=30, title=f"Distribusi {selected_num_col}")
                st.plotly_chart(fig)
        else:
            st.warning("Dataset tidak memiliki kolom numerik untuk analisis distribusi.")

        # Visualisasi Distribusi untuk Kolom Kategorikal
        if len(categorical_cols) > 0:
            selected_cat_col = st.selectbox("Pilih Kolom Kategorikal untuk Visualisasi Frekuensi:", categorical_cols)
            if selected_cat_col:
                st.write(f"### 🔠 Distribusi Kolom: {selected_cat_col}")
                fig = px.bar(data[selected_cat_col].value_counts().reset_index(), 
                             x='index', y=selected_cat_col, 
                             labels={'index': selected_cat_col, selected_cat_col: 'Frequency'},
                             title=f"Distribusi {selected_cat_col}")
                st.plotly_chart(fig)
        else:
            st.warning("Dataset tidak memiliki kolom kategorikal untuk analisis distribusi.")

        # Heatmap Korelasi
        if len(numerical_cols) > 1:
            st.write("### 🌡️ Heatmap Korelasi")
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig_corr)
        else:
            st.warning("Dataset tidak memiliki cukup kolom numerik untuk menampilkan heatmap korelasi.")

        # Scatter Matrix
        if len(numerical_cols) > 1:
            st.write("### 🔍 Scatter Matrix")
            scatter_fig = px.scatter_matrix(
                data, 
                dimensions=numerical_cols, 
                title="Scatter Matrix",
                labels={col: col for col in numerical_cols}
            )
            st.plotly_chart(scatter_fig)

elif option == "📈 Modeling":
    st.title("📈 Modeling")
    st.write("### 🧠 Latih Model Machine Learning Anda")

    # Memuat dataset untuk modeling
    data = load_dataset("Modeling")
    if data is not None:
        st.write("### 📋 Dataset Overview")
        st.dataframe(data.head())

        try:
            # Pilih kolom target
            target = st.selectbox("🎯 Pilih Target Variable:", data.columns)
            features = data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = data[target]

            # Validasi dataset
            if X.empty or y.empty:
                st.error("Dataset tidak memiliki fitur numerik atau target yang valid.")
                st.stop()

            # Pembagian dataset
            test_size = st.slider("📊 Pilih Proporsi Data Uji:", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Pilih algoritma
            model_option = st.selectbox("📚 Pilih Algoritma Machine Learning:", 
                                        ["Logistic Regression", "Random Forest"])

            if model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_option == "Random Forest":
                n_estimators = st.slider("🔢 Jumlah Pohon dalam Random Forest:", 
                                         min_value=10, max_value=200, value=100, step=10)
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            # Latih model
            st.write("### 🔄 Melatih Model...")
            model.fit(X_train, y_train)
            st.success(f"Model {model_option} berhasil dilatih!")

            # Evaluasi model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write("### 📊 Evaluasi Model")
            st.metric("🎯 Akurasi", f"{accuracy:.2f}")
            st.text("📋 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Feature importance untuk Random Forest
            if model_option == "Random Forest":
                feature_importance = model.feature_importances_
                feature_names = X.columns
                st.write("### 🔍 Feature Importance")
                fig = px.bar(x=feature_names, y=feature_importance, title="Feature Importance")
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data atau model: {e}")
