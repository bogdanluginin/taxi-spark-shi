import os
import sys
from pyspark.sql import SparkSession

def main():
    # Set Python executable path for both driver and workers
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    # Ініціалізація SparkSession (точки входу)
    spark = SparkSession.builder \
        .appName("Spark Setup Test") \
        .master("local[*]") \
        .getOrCreate()

    print(f"--- Успіх! Версія Spark: {spark.version} ---")

    # Створення тестового DataFrame
    data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
    columns = ["Name", "Age"]

    df = spark.createDataFrame(data, columns)

    # Застосування методу .show()
    print("Тестовий DataFrame успішно створено:")
    df.show()

    # Зупинка сесії
    spark.stop()

if __name__ == "__main__":
    main()