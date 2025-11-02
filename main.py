import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from data_loader import load_data
# --- ІМПОРТУЄМО ВСІ НАШІ ФУНКЦІЇ ---
from transformer import (
    run_initial_analysis,
    get_question_1,
    get_question_2,
    get_question_3,
    get_question_4,
    get_question_5,
    get_question_6
)


def main():
    # Налаштування Python, щоб уникнути помилки версій
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

    # Ініціалізація Spark з вашими налаштуваннями
    spark = SparkSession.builder \
        .appName("NYC Taxi Transformation Stage") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    print(f"--- Успіх! Версія Spark: {spark.version} ---")

    file_path = "/Users/azzasel/Documents/2025 lab /trip_data_10.csv"
    # Папка для збереження результатів (має бути у .gitignore)
    results_path = "results"

    try:
        # 1. ЕТАП ВИДОБУВАННЯ
        print(f"--- Завантаження даних з {file_path} ---")
        taxi_df = load_data(spark, file_path)

        # КЕШУЄМО DataFrame, оскільки ми будемо
        # виконувати 7 запитів до нього (1 аналіз + 6 питань)
        taxi_df.cache()

        # 2. ЕТАП ТРАНСФОРМАЦІЇ (Завдання 1 і 2)
        run_initial_analysis(taxi_df)

        # 3. ЕТАП ТРАНСФОРМАЦІЇ (Завдання 3: Бізнес-питання)
        # 4. ЕТАП ЗАПИСУ РЕЗУЛЬТАТІВ
        print("\n--- ЗАВДАННЯ 3: Виконання 6 бізнес-питань ---")

        # --- Питання 1 + Збереження ---
        q1_result = get_question_1(taxi_df)
        q1_result.write.mode("overwrite").option("header", "true").csv(f"{results_path}/q1_top_rush_hour_pickups")

        # --- Питання 2 + Збереження ---
        q2_result = get_question_2(taxi_df)
        q2_result.write.mode("overwrite").option("header", "true").csv(f"{results_path}/q2_avg_stats_by_vendor")

        # --- Питання 3 + Збереження ---
        q3_result = get_question_3(taxi_df)
        q3_result.write.mode("overwrite").option("header", "true").csv(f"{results_path}/q3_zero_trips_sample")

        # --- Питання 4 + Збереження ---
        q4_result = get_question_4(taxi_df, spark)
        q4_result.write.mode("overwrite").option("header", "true").csv(f"{results_path}/q4_distance_by_rate_type")

        # --- Питання 5 + Збереження ---
        q5_result = get_question_5(taxi_df)
        q5_result.write.mode("overwrite").option("header", "true").csv(f"{results_path}/q5_top_3_trips_per_day")

        # --- Питання 6 + Збереження ---
        q6_result = get_question_6(taxi_df, spark)
        q6_result.write.mode("overwrite").option("header", "true").csv(f"{results_path}/q6_sliding_avg_distance")

        print("\n--- Усі 6 бізнес-питань виконано та збережено у папку 'results' ---")

        # Звільняємо кеш
        taxi_df.unpersist()

    except AnalysisException as e:
        print(f"--- ПОМИЛКА: Файл не знайдено або шлях невірний ---")
        print(e)
    except Exception as e:
        print(f"--- Виникла інша помилка: ---")
        print(e)
        import traceback
        traceback.print_exc()

    finally:
        print("--- Зупинка сесії Spark ---")
        try:
            # Ваша покращена перевірка, чи сесія активна
            if spark is not None and hasattr(spark, '_jvm') and spark._jvm is not None:
                spark.stop()
                print("--- Spark сесію успішно зупинено ---")
            else:
                print("--- Spark сесія вже була зупинена ---")
        except Exception as stop_error:
            print(f"--- Помилка при зупинці Spark (можна ігнорувати): {stop_error} ---")


if __name__ == "__main__":
    main()