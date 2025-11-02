import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from data_loader import load_data  # <--- ІМПОРТУЄМО НАШУ НОВУ ФУНКЦІЮ


def main():
    # Встановлюємо шлях до Python
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

    # Ініціалізація Spark
    spark = SparkSession.builder \
        .appName("NYC Taxi Data Analysis") \
        .master("local[*]") \
        .getOrCreate()

    print(f"--- Успіх! Версія Spark: {spark.version} ---")

    # Шлях до файлу
    file_path = "/Users/azzasel/Documents/2025 lab /trip_data_10.csv"

    try:
        # Викликаємо нашу функцію з модуля
        print(f"--- Завантаження даних з {file_path} ---")
        taxi_df = load_data(spark, file_path)

        # Перевірка
        print("--- Схема даних з data_loader: ---")
        taxi_df.printSchema()

        print("--- Дані успішно завантажено: ---")
        taxi_df.show(5)

    except AnalysisException as e:
        print(f"--- ПОМИЛКА: Файл не знайдено або шлях невірний ---")
        print(e)
    except Exception as e:
        print(f"--- Виникла інша помилка: ---")
        print(e)

    finally:
        print("--- Зупинка сесії Spark ---")
        spark.stop()


if __name__ == "__main__":
    main()