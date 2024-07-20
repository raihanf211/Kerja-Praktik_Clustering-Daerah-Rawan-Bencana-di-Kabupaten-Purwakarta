import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import folium
import seaborn as sns
from streamlit_option_menu import option_menu


def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


local_css("style.css")

# Load data
file_path = "DatasetDRB_peta.csv"
df = pd.read_csv(file_path)

# Sidebar
st.sidebar.title("Pengaturan Klaster")
num_clusters = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=[
            "Data Asli",
            "Tabel Klaster",
            "Visualisasi Data",
            "Peta Folium",
            "Silhoute Score",
        ],
        icons=["journal-code", "journal-check", "graph-up", "pin-map", "activity"],
        menu_icon="cast",
        default_index=0,
    )
    st.sidebar.markdown("----")

# Cluster the data
features = ["Longsor", "Gempa"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Kolom untuk pengelompokan (Longsor dan Gempa)
kolom_pengelompokan = ["Longsor", "Gempa"]

# Kategorikan klaster berdasarkan rentang kerawanan wilayah
Vulnerability_ranges = [-1, 5, 9, np.inf]

df["Vulnerability Category"] = pd.cut(
    df[kolom_pengelompokan].mean(axis=1),
    bins=Vulnerability_ranges,
    labels=["Rendah", "Sedang", "Tinggi"],
)

# Metode Elbow untuk menentukan jumlah klaster yang optimal
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, random_state=0, n_init=10, max_iter=50
    )  # Set n_init explicitly
    kmeans.fit(X_scaled)  # Use the original scaled data, not df_normalized
    inertia_values.append(kmeans.inertia_)

# Menampilkan plot Metode Elbow
fig, ax = plt.subplots()
ax.plot(range(1, 11), inertia_values, marker="o")
ax.set_xlabel("Jumlah Klaster")
ax.set_ylabel("Inertia (Within-cluster Sum of Squares)")
st.sidebar.pyplot(fig)

# Menampilkan informasi Metode Elbow
st.sidebar.write("### Informasi Metode Elbow:")
st.sidebar.write(
    "Metode Elbow membantu menentukan jumlah klaster optimal dengan melihat titik di mana penurunan inersia tidak lagi signifikan."
)

# Konten utama
st.title("Pengelompokkan Daerah Rawan Bencana di Kabupaten Purwakarta")
st.markdown("----")

# Data asli
if selected == "Data Asli":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")
    st.write(df)

# Data klaster
elif selected == "Tabel Klaster":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")

    # Categorize klaster berdasarkan "Vulnerability Category"
    df["Vulnerability Category"] = pd.cut(
        df[kolom_pengelompokan].mean(axis=1),
        bins=Vulnerability_ranges,
        labels=["Rendah", "Sedang", "Tinggi"],
    )

    # Display data for all clusters with their "Vulnerability Category"
    for cluster_number in range(num_clusters):
        st.write(
            f"### Data untuk Klaster {cluster_number + 0} ({df.loc[df['Cluster'] == cluster_number, 'Vulnerability Category'].iloc[0]})"
        )
        st.write(df[df["Cluster"] == cluster_number])

# Silhouette Score
elif selected == "Silhoute Score":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")

    # Create a temporary DataFrame without "Vulnerability Category" for Silhouette Score calculation
    data_for_silhouette = df.drop(
        ["Kecamatan", "Latitude", "Longitude", "Cluster", "Vulnerability Category"],
        axis=1,
    )

    scaler = StandardScaler()
    normalized_data_silhouette = scaler.fit_transform(data_for_silhouette)
    df_normalized_silhouette = pd.DataFrame(
        normalized_data_silhouette, columns=data_for_silhouette.columns
    )
    df_normalized_silhouette[["Kecamatan", "Latitude", "Longitude", "Cluster"]] = df[
        ["Kecamatan", "Latitude", "Longitude", "Cluster"]
    ]
    X_scaled_silhouette = df_normalized_silhouette.drop(
        ["Kecamatan", "Latitude", "Longitude", "Cluster"], axis=1
    )

    silhouette_scores = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    fig, ax = plt.subplots()  # Create a subplot

    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=50)
        cluster_labels = kmeans.fit_predict(X_scaled_silhouette)

        if len(set(cluster_labels)) > 1:  # Check if there are at least 2 unique labels
            silhouette_avg = silhouette_score(X_scaled_silhouette, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(None)

    ax.plot(range_n_clusters, silhouette_scores, marker="o")
    ax.set_title("Silhouette Score for Different Numbers of Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")

    # Add silhouette scores below the plot
    for i, score in enumerate(silhouette_scores):
        if score is not None:
            ax.text(
                range_n_clusters[i], score, f"{score:.2f}", ha="center", va="bottom"
            )
            st.write(
                f"For n_clusters = {range_n_clusters[i]}, the silhouette score is {score:.2f}"
            )

    st.pyplot(fig)

# Visualisasi Data
elif selected == "Visualisasi Data":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")

    # Scatter plot antara Latitude dan Longitude
    st.write("### Scatter Plot Latitude vs Longitude:")
    fig, ax = plt.subplots()
    ax.scatter(df["Longitude"], df["Latitude"])
    ax.set_title("Scatter Plot Latitude vs Longitude")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

    st.markdown("---")

    # Scatter Plot Longitude vs Latitude dengan Warna berdasarkan Longsor
    st.write("### Scatter Plot Longitude vs Latitude dengan Warna berdasarkan Longsor:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Longitude",
        y="Latitude",
        hue="Longsor",
        palette="coolwarm",
        s=100,
        ax=ax,
    )
    ax.set_title("Scatter Plot Longitude vs Latitude dengan Warna berdasarkan Longsor")
    st.pyplot(fig)

    st.markdown("---")

    # Scatter Plot Longitude vs Latitude dengan Warna berdasarkan Gempa
    st.write("### Scatter Plot Longitude vs Latitude dengan Warna berdasarkan Gempa:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Longitude",
        y="Latitude",
        hue="Gempa",
        palette="coolwarm",
        s=100,
        ax=ax,
    )
    ax.set_title("Scatter Plot Longitude vs Latitude dengan Warna berdasarkan Gempa")
    st.pyplot(fig)


elif selected == "Peta Folium":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")

    # Membuat peta dengan lokasi rata-rata latitude dan longitude
    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12
    )

    # Menambahkan marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Menambahkan penanda untuk setiap klaster
    for i, row in df.iterrows():
        vulnerability_category_info = f"Klaster {row['Cluster'] + 0}"

        # Menyesuaikan warna ikon berdasarkan klaster
        icon_color = (
            "green"
            if row["Cluster"] == 0
            else "orange"
            if row["Cluster"] == 1
            else "red"
        )

        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            popup=f"{row['Kecamatan']}<br>{vulnerability_category_info}",
            icon=folium.Icon(color=icon_color),
        ).add_to(marker_cluster)

    # Menambahkan peta ke Streamlit
    folium_static(m)

# Menyimpan hasil klaster untuk setiap klaster
cluster_results = []
for cluster_id in range(num_clusters):
    cluster_data = df[df["Cluster"] == cluster_id]

    # Menggabungkan Kecamatan menjadi satu teks
    kecamatan_text = ", ".join(cluster_data["Kecamatan"].tolist())

    # Menentukan tingkat rawan
    if not cluster_data.empty:
        vulnerability_category = cluster_data["Vulnerability Category"].iloc[0]

        cluster_results.append(
            {f"Klaster {cluster_id + 0} ({vulnerability_category})": kecamatan_text}
        )
st.sidebar.markdown("---")
st.sidebar.write("### Kesimpulan:")
# Menampilkan kesimpulan
for result in cluster_results:
    for key, value in result.items():
        st.sidebar.write(f"{key}: {value}")


# Menambahkan penjelasan kesimpulan berdasarkan hasil klasterisasi, nilai Silhouette Score, dll.
st.sidebar.markdown("---")
st.sidebar.write("### Informasi Tambahan:")
st.sidebar.write(
    "Analisis ini didasarkan pada data di website Open Data Purwakarta dengan link https://data.purwakartakab.go.id yang di miliki oleh Diskominfo dan dikelola oleh Bidang Statistik "
    "informasi yang didapatkan adalah jumlah desa kelurahan yang mengalami bencana alam Tanah longsor dan Gempa bumi menurut Kecamatan di Kabupaten Purwakarta, Data yang dipakai adalah dari tahun 2014, 2018, dan 2020 dengan jumlah dataset 102 record, "
    "data bersumber dari Dinas Pemadam Kebakaran dan Penanggulangan Bencana."
)

st.sidebar.write(
    "Selain itu, Metode Elbow digunakan untuk menentukan jumlah klaster optimal. "
    "Jumlah klaster yang dipilih didasarkan pada poin di mana penurunan inersia tidak lagi signifikan, "
    "sehingga memberikan klasterisasi yang baik."
)
