from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def run_initial_analysis(df: DataFrame):
    """
    Виконує Завдання 1 (Загальна інфо) та
    Завдання 2 (Статистика) з Етапу трансформації.
    """

    # --------------------------------------------------
    # ЗАВДАННЯ 1: Загальна інформація про набір даних
    # --------------------------------------------------
    print("\n--- ЗАВДАННЯ 1: Загальна інформація про набір даних ---")

    print("Схема даних (Структура):")
    df.printSchema()

    # Отримуємо кількість колонок без матеріалізації даних
    col_count = len(df.columns)

    # Підраховуємо рядки (це може бути повільно для великих файлів)
    print("\nПідрахунок кількості рядків...")
    row_count = df.count()

    print(f"\nКількість рядків (поїздок): {row_count}")
    print(f"Кількість колонок (характеристик): {col_count}")

    print("\nОпис:")
    print(f"Набір даних містить {row_count} записів про поїздки таксі,")
    print(f"кожна з яких описана {col_count} параметрами, що включають")
    print("унікальні ідентифікатори, час, координати та деталі поїздки.")

    # --------------------------------------------------
    # ЗАВДАННЯ 2: Статистика щодо числових стовпців
    # --------------------------------------------------
    print("\n--- ЗАВДАННЯ 2: Статистика щодо числових стовпців ---")

    # Автоматично обираємо всі числові колонки для аналізу
    numeric_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, (DoubleType, IntegerType))]

    print("Вибрані числові колонки для аналізу:", numeric_cols)

    # .describe() рахує count, mean, stddev, min, max
    # Використовуємо лише необхідні колонки для економії пам'яті
    if numeric_cols:
        df.select(numeric_cols).describe().show()
    else:
        print("Числові колонки не знайдено у DataFrame")


# --------------------------------------------------
# ЗАВДАННЯ 3: 6 Бізнес-питань
# --------------------------------------------------

def get_question_1(df: DataFrame) -> DataFrame:
    """
    [Filter + GroupBy]
    Які 5 найпопулярніших точок посадки (за координатами)
    у 'годину пік' (з 17:00 до 19:00)?
    """
    print("\n--- БІЗНЕС-ПИТАННЯ 1: Топ-5 точок посадки у 'годину пік' ---")

    q1_df = df.filter(F.hour(df["pickup_datetime"]).between(17, 19)) \
        .groupBy("pickup_longitude", "pickup_latitude") \
        .count() \
        .orderBy(F.col("count").desc())

    q1_df.show(5)
    # Зберігаємо 100 найпопулярніших, а не всі мільйони
    return q1_df.limit(100)


def get_question_2(df: DataFrame) -> DataFrame:
    """
    [GroupBy]
    Яка середня тривалість поїздки (в сек) та середня дистанція
    для кожного `vendor_id`?
    """
    print("\n--- БІЗНЕС-ПИТАННЯ 2: Середня тривалість та дистанція для vendor_id ---")

    q2_df = df.groupBy("vendor_id") \
        .agg(
        F.round(F.avg("trip_time_in_secs"), 2).alias("avg_duration_secs"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles")
    )

    q2_df.show()
    return q2_df


def get_question_3(df: DataFrame) -> DataFrame:
    """
    [Filter]
    Скільки було "нульових" поїздок (де пасажири = 0 АБО дистанція = 0)?
    """
    print("\n--- БІЗНЕС-ПИТАННЯ 3: Кількість 'нульових' поїздок ---")

    # Ми знаємо з нашого аналізу, що такі поїздки існують
    q3_df = df.filter(
        (F.col("passenger_count") == 0) | (F.col("trip_distance") == 0)
    )

    count = q3_df.count()
    print(f"Знайдено {count} 'нульових' поїздок.")

    # Повертаємо DataFrame, щоб зберегти приклад "брудних" даних
    return q3_df.limit(1000)


def get_question_4(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    [Join + GroupBy]
    Яка загальна дистанція (як міра прибутку) для кожного типу тарифу?
    """
    print("\n--- БІЗНЕС-ПИТАННЯ 4: Загальна дистанція за типом тарифу ---")

    # Створюємо довідник (Lookup DataFrame)
    rate_codes_data = [
        (1, "Standard"), (2, "JFK"), (3, "Newark"),
        (4, "Nassau/Westchester"), (5, "Negotiated"), (6, "Group")
    ]
    rate_codes_df = spark.createDataFrame(rate_codes_data, ["rate_code", "rate_name"])

    # [Join + GroupBy]
    q4_df = df.join(rate_codes_df, "rate_code") \
        .groupBy("rate_name") \
        .agg(F.round(F.sum("trip_distance"), 2).alias("total_distance")) \
        .orderBy(F.col("total_distance").desc())

    q4_df.show()
    return q4_df


def get_question_5(df: DataFrame) -> DataFrame:
    """
    [Window Function]
    Знайти 3 найдовші (за trip_distance) поїздки для *кожного* дня місяця.
    """
    print("\n--- БІЗНЕС-ПИТАННЯ 5: Топ-3 найдовших поїздок за кожен день ---")

    # [Window Function]
    window_spec = Window.partitionBy(F.dayofmonth("pickup_datetime")) \
        .orderBy(F.col("trip_distance").desc())

    q5_df = df.withColumn("rank", F.rank().over(window_spec)) \
        .filter(F.col("rank") <= 3) \
        .select("pickup_datetime", "trip_distance", "rank") \
        .orderBy("pickup_datetime", "rank")

    q5_df.show(10)
    return q5_df


def get_question_6(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    [Window Function + Join + Filter]
    Яка 'ковзна середня' дистанція поїздки за останні 2 поїздки
    для кожного водія, *крім* тих, хто у 'чорному списку'?
    """
    print("\n--- БІЗНЕС-ПИТАННЯ 6: Ковзна середня дистанція (крім чорного списку) ---")

    # 1. Створюємо фальшивий "чорний список" водіїв
    # (Беремо ID з попередніх виводів .show())
    blacklist_data = [("E48B185060FB0FF49CF233E3F3980A18",), ("1D10D8AC5B07D808643B6A96243B4896",)]
    blacklist_df = spark.createDataFrame(blacklist_data, ["hack_license"])

    # 2. [Filter + Join] Використовуємо 'left_anti' join, щоб відфільтрувати водіїв
    filtered_drivers_df = df.join(blacklist_df, "hack_license", "left_anti")

    # 3. [Window Function]
    window_spec = Window.partitionBy("hack_license") \
        .orderBy("pickup_datetime") \
        .rowsBetween(-1, 0)  # Поточна + 1 попередня = 2 поїздки

    q6_df = filtered_drivers_df.withColumn(
        "sliding_avg_distance",
        F.round(F.avg("trip_distance").over(window_spec), 2)
    )

    q6_df.select("hack_license", "pickup_datetime", "trip_distance", "sliding_avg_distance").show(10)
    # Зберігаємо приклад, а не 15 мільйонів рядків
    return q6_df.limit(1000)