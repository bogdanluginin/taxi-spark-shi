import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # <--- НОВИЙ ІМПОРТ
import glob
import os

# --- Налаштування шляхів ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"


# --- Функція 1: Пошук CSV (без змін) ---
def find_csv_file(folder_name: str) -> str:
    search_path = os.path.join(RESULTS_DIR, folder_name, "*.csv")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"Не знайдено CSV файлів у папці {search_path}")
    return files[0]


# --- Функція 2: Графік Q2 (matplotlib, без змін) ---
def plot_q2_avg_stats_by_vendor():
    try:
        csv_path = find_csv_file("q2_avg_stats_by_vendor")
        df = pd.read_csv(csv_path)
        print(f"\nДані для Q2 (статистика за vendor) успішно завантажено:\n{df}")
        df.plot(kind='bar', x='vendor_id', y=['avg_duration_secs', 'avg_distance_miles'],
                secondary_y=['avg_distance_miles'], title='Q2: Середня тривалість та дистанція за Vendor ID')
        plt.xlabel("ID Постачальника")
        plt.ylabel("Середня тривалість (секунди)")
        save_path = os.path.join(PLOTS_DIR, "q2_avg_stats_by_vendor.png")
        plt.savefig(save_path)
        print(f"Графік Q2 успішно збережено у {save_path}")
        plt.close()
    except Exception as e:
        print(f"Помилка при побудові Q2: {e}")


# --- Функція 3: Графік Q4 (matplotlib, без змін) ---
def plot_q4_distance_by_rate_type():
    try:
        csv_path = find_csv_file("q4_distance_by_rate_type")
        df = pd.read_csv(csv_path)
        print(f"\nДані для Q4 (дистанція за тарифом) успішно завантажено:\n{df}")
        df = df.sort_values(by='total_distance', ascending=True)
        df.plot(kind='barh', x='rate_name', y='total_distance',
                title='Q4: Загальна дистанція за типом тарифу', legend=False)
        plt.xlabel("Загальна дистанція (млн. миль)")
        plt.ylabel("Тип тарифу")
        save_path = os.path.join(PLOTS_DIR, "q4_distance_by_rate_type.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Графік Q4 успішно збережено у {save_path}")
        plt.close()
    except Exception as e:
        print(f"Помилка при побудові Q4: {e}")


# --- НОВІ ФУНКЦІЇ SEABORN ---

def plot_distributions(sample_df: pd.DataFrame):
    """
    Гістограма (seaborn) для тривалості та дистанції.
    """
    print("\n--- Побудова графіків розподілу (Seaborn) ---")

    # 1. Графік для trip_time_in_secs
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_df['trip_time_in_secs'], kde=True, bins=50)
    plt.title('Розподіл тривалості поїздки (trip_time_in_secs)')
    plt.xlabel('Тривалість (секунди)')
    plt.ylabel('Кількість поїздок')
    # Зберігаємо
    save_path = os.path.join(PLOTS_DIR, "s1_distribution_time.png")
    plt.savefig(save_path)
    print(f"Графік розподілу тривалості збережено у {save_path}")
    plt.close()


def plot_scatter(sample_df: pd.DataFrame):
    """
    Діаграма розсіювання (seaborn) для дистанції vs тривалість.
    """
    print("\n--- Побудова діаграми розсіювання (Seaborn) ---")

    # 1. Графік для trip_distance vs trip_time_in_secs
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample_df,
        x='trip_distance',
        y='trip_time_in_secs',
        alpha=0.5  # Прозорість
    )
    plt.title('Залежність тривалості від дистанції')
    plt.xlabel('Дистанція (милі)')
    plt.ylabel('Тривалість (секунди)')

    # 2. Зберігаємо
    save_path = os.path.join(PLOTS_DIR, "s2_scatter_distance_time.png")
    plt.savefig(save_path)
    print(f"Графік розсіювання збережено у {save_path}")
    plt.close()


def plot_correlation_heatmap(sample_df: pd.DataFrame):
    """
    Теплова карта кореляції (seaborn).
    """
    print("\n--- Побудова теплової карти кореляції (Seaborn) ---")

    # 1. Обираємо числові колонки для кореляції
    cols_to_correlate = [
        'trip_distance', 'trip_time_in_secs',
        'passenger_count', 'pickup_longitude',
        'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
    ]
    corr_matrix = sample_df[cols_to_correlate].corr()

    # 2. Створюємо графік
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,  # Показувати цифри
        fmt=".2f",  # 2 знаки після коми
        cmap='coolwarm'  # Кольорова схема
    )
    plt.title('Теплова карта кореляції ознак')

    # 3. Зберігаємо
    save_path = os.path.join(PLOTS_DIR, "s3_correlation_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Теплову карту збережено у {save_path}")
    plt.close()


def main():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Створено папку {PLOTS_DIR}")

    # --- 1. Старі графіки (matplotlib) ---
    plot_q2_avg_stats_by_vendor()
    plot_q4_distance_by_rate_type()

    # --- 2. Нові графіки (seaborn) ---
    try:
        # Знаходимо наш новий файл зі зразком
        sample_csv_path = find_csv_file("cleaned_data_sample")
        sample_df = pd.read_csv(sample_csv_path)

        # Обмежуємо дані, щоб уникнути занадто довгих "хвостів" на графіках
        sample_df_filtered = sample_df[
            (sample_df['trip_time_in_secs'] < 5000) &
            (sample_df['trip_distance'] < 60)
            ]

        plot_distributions(sample_df_filtered)
        plot_scatter(sample_df_filtered)
        plot_correlation_heatmap(sample_df)  # Кореляцію рахуємо на всіх даних

    except FileNotFoundError as e:
        print(f"ПОМИЛКА: {e}. Спочатку запустіть 'run_ml_analysis.py', щоб згенерувати 'cleaned_data_sample'.")
    except Exception as e:
        print(f"Виникла загальна помилка: {e}")


if __name__ == "__main__":
    main()