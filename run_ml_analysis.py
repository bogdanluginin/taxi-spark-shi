import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import DoubleType, IntegerType
from data_loader import load_data
# --- ІМПОРТУЄМО ВСІ ФУНКЦІЇ ДЛЯ ОБОХ ЕТАПІВ ML ---
from transformer import (
    preprocess_data_for_ml,
    create_features_for_regression,
    train_regression_models,
    create_features_for_classification,
    train_classification_models
)


def main():
    # Налаштування Python
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

    # Ініціалізація Spark
    spark = SparkSession.builder \
        .appName("NYC Taxi ML Analysis Stage") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    print(f"--- Успіх! Версія Spark: {spark.version} ---")

    file_path = "/Users/azzasel/Documents/2025 lab /trip_data_10.csv"

    try:
        # 1. Завантаження даних
        print(f"--- Завантаження даних з {file_path} ---")
        taxi_df = load_data(spark, file_path)

        # 2. Попередня обробка
        cleaned_df = preprocess_data_for_ml(taxi_df)

        # Кешуємо очищені дані, оскільки ми будемо використовувати їх двічі
        cleaned_df.cache()
        print("\n--- Очищені дані збережено в кеш ---")

        # =================================================
        # ЗАВДАННЯ РЕГРЕСІЇ
        # =================================================

        # 3. Інженерія ознак для Регресії
        regression_data = create_features_for_regression(cleaned_df)
        regression_data.cache()

        # 4. Навчання та Оцінка Моделей Регресії
        train_regression_models(regression_data)

        regression_data.unpersist()
        print("\n--- Етап Регресії завершено ---")

        # =================================================
        # ЗАВДДАННЯ КЛАСИФІКАЦІЇ
        # =================================================

        # 5. Інженерія ознак для Класифікації
        classification_data = create_features_for_classification(cleaned_df)
        classification_data.cache()

        # 6. Навчання та Оцінка Моделей Класифікації
        train_classification_models(classification_data)

        classification_data.unpersist()
        print("\n--- Етап Класифікації завершено ---")

        cleaned_df.unpersist()  # Звільняємо головний кеш

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
            if spark is not None and hasattr(spark, '_jvm') and spark._jvm is not None:
                spark.stop()
                print("--- Spark сесію успішно зупинено ---")
            else:
                print("--- Spark сесія вже була зупинена ---")
        except Exception as stop_error:
            print(f"--- Помилка при зупинці Spark (можна ігнорувати): {stop_error} ---")


if __name__ == "__main__":
    main()