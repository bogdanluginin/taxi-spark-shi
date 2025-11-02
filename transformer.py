from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.storagelevel import StorageLevel
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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


# --- ДОДАЙТЕ ЦІ ІМПОРТИ НА ПОЧАТОК transformer.py ---
# (F та Window у вас вже мають бути, додайте StorageLevel)



# ... (тут ваші 6 бізнес-питань) ...


# --------------------------------------------------
# ЕТАП АНАЛІЗУ (Крок 2: Попередня обробка)
# --------------------------------------------------

def preprocess_data_for_ml(df: DataFrame) -> DataFrame:
    """
    Виконує попередню обробку даних для ML:
    1. Видаляє пропуски (nulls).
    2. Фільтрує "нульові" поїздки (дистанція, час, пасажири).
    3. Фільтрує аномальні тарифи (залишає 1-6).
    4. Фільтрує аномальні координати (залишає тільки NYC).
    """
    print("\n--- ЕТАП АНАЛІЗУ (Крок 2): Початок попередньої обробки даних ---")

    # Використовуємо .persist() для кешування
    df.persist(StorageLevel.MEMORY_AND_DISK)

    start_count = df.count()
    print(f"Початкова кількість рядків: {start_count}")

    # 1. Видаляємо пропуски (nulls)
    df_cleaned = df.na.drop()
    count_after_nulls = df_cleaned.count()
    print(
        f"Рядків після видалення пропусків (nulls): {count_after_nulls} (Видалено: {start_count - count_after_nulls})")

    # 2. Фільтрація нульових значень
    df_cleaned = df_cleaned.filter(
        (F.col("trip_distance") > 0) &
        (F.col("passenger_count") > 0) &
        (F.col("trip_time_in_secs") > 0)
    )
    count_after_zeros = df_cleaned.count()
    print(
        f"Рядків після видалення 'нульових' поїздок: {count_after_zeros} (Видалено: {count_after_nulls - count_after_zeros})")

    # 3. Фільтрація аномальних тарифів
    df_cleaned = df_cleaned.filter(F.col("rate_code").between(1, 6))
    count_after_rates = df_cleaned.count()
    print(
        f"Рядків після фільтрації тарифів (1-6): {count_after_rates} (Видалено: {count_after_zeros - count_after_rates})")

    # 4. Фільтрація координат (NYC)
    df_cleaned = df_cleaned.filter(
        (F.col("pickup_latitude").between(40.49, 40.91)) &
        (F.col("pickup_longitude").between(-74.25, -73.70)) &
        (F.col("dropoff_latitude").between(40.49, 40.91)) &
        (F.col("dropoff_longitude").between(-74.25, -73.70))
    )

    end_count = df_cleaned.count()
    print(f"Рядків після фільтрації координат (фінальна кількість): {end_count}")
    print(f"\n--- Всього видалено 'брудних' рядків: {start_count - end_count} ---")

    df.unpersist()  # Звільняємо кеш "брудного" DF

    # Повертаємо ОЧИЩЕНИЙ DataFrame
    return df_cleaned


# --------------------------------------------------
# ЕТАП АНАЛІЗУ (Крок 2.5: Інженерія ознак для регресії)
# --------------------------------------------------

def create_features_for_regression(df: DataFrame) -> DataFrame:
    """
    Готує дані для моделі регресії (прогнозування trip_time_in_secs).
    1. Створює нові ознаки з дати/часу.
    2. Перейменовує цільову змінну на 'label'.
    3. Збирає всі ознаки у 'features' вектор.
    """
    print("--- Початок інженерії ознак для регресії ---")

    # 1. Створюємо нові ознаки
    df_with_features = df.withColumn("pickup_hour", hour(col("pickup_datetime"))) \
        .withColumn("day_of_week", dayofweek(col("pickup_datetime")))

    # 2. Перейменовуємо цільову змінну (це вимога Spark ML)
    df_with_label = df_with_features.withColumnRenamed("trip_time_in_secs", "label")

    # 3. Збираємо всі ознаки у вектор
    # Наші ознаки: 'trip_distance', 'passenger_count', 'pickup_hour', 'day_of_week'
    assembler = VectorAssembler(
        inputCols=["trip_distance", "passenger_count", "pickup_hour", "day_of_week"],
        outputCol="features"
    )

    ml_ready_df = assembler.transform(df_with_label)

    # Залишаємо тільки ті колонки, які потрібні моделі
    final_df = ml_ready_df.select("features", "label")

    return final_df


# --------------------------------------------------
# ЕТАП АНАЛІЗУ (Крок 3, 4, 5, 6: Моделі Регресії)
# --------------------------------------------------

def train_regression_models(data: DataFrame):
    """
    Навчає, тестує та оцінює 3 моделі регресії.
    """
    print("\n--- Початок навчання моделей регресії ---")

    # 1. Розділяємо дані на тренувальний та тестовий набори
    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Тренувальний набір: {train_data.count()} рядків")
    print(f"Тестовий набір: {test_data.count()} рядків")

    # 2. Визначаємо наші 3 моделі
    lr = LinearRegression(featuresCol='features', labelCol='label')
    dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')
    rf = RandomForestRegressor(featuresCol='features', labelCol='label', numTrees=20, maxDepth=5)  # Полегшена версія

    models = [lr, dt, rf]
    model_names = ["Linear Regression", "Decision Tree", "Random Forest"]

    # 3. Визначаємо оцінювачі (Evaluators)
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    print("\n--- Порівняння результатів (Завдання 6) ---")
    print("Модель | RMSE | R² (R-squared)")
    print("-" * 30)

    # 4. Навчаємо, прогнозуємо та оцінюємо кожну модель
    for model, name in zip(models, model_names):
        try:
            # Навчаємо модель
            print(f"Навчання {name}...")
            trained_model = model.fit(train_data)

            # Робимо прогнози
            predictions = trained_model.transform(test_data)

            # Оцінюємо якість
            rmse = evaluator_rmse.evaluate(predictions)
            r2 = evaluator_r2.evaluate(predictions)

            print(f"{name} | {rmse:.2f} | {r2:.2f}")

        except Exception as e:
            print(f"Помилка під час навчання {name}: {e}")


# --------------------------------------------------
# ЕТАП АНАЛІЗУ (Крок 2.5: Інженерія ознак для класифікації)
# --------------------------------------------------

def create_features_for_classification(df: DataFrame) -> DataFrame:
    """
    Готує дані для моделі класифікації (прогнозування rate_code).
    1. Конвертує 'rate_code' у 0-індексовану 'label'.
    2. Збирає ознаки у 'features' вектор.
    """
    print("--- Початок інженерії ознак для класифікації ---")

    # 1. Створюємо ознаки
    # Ми будемо використовувати дистанцію та координати для прогнозування типу тарифу
    feature_cols = [
        "trip_distance", "passenger_count",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude"
    ]

    # 2. Збираємо ознаки у вектор
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unindexed")
    df_assembled = assembler.transform(df)

    # 3. Конвертуємо 'rate_code' (який 1, 2, 3...) у 0-індексовану 'label' (0, 1, 2...)
    # Це вимога Spark ML для класифікаторів
    indexer = StringIndexer(inputCol="rate_code", outputCol="label")
    ml_ready_df = indexer.fit(df_assembled).transform(df_assembled)

    final_df = ml_ready_df.select("features_unindexed", "label") \
        .withColumnRenamed("features_unindexed", "features")

    return final_df


# --------------------------------------------------
# ЕТАП АНАЛІЗУ (Крок 3, 4, 5, 6: Моделі Класифікації)
# --------------------------------------------------

def train_classification_models(data: DataFrame):
    """
    Навчає, тестує та оцінює 3 моделі класифікації.
    """
    print("\n--- Початок навчання моделей класифікації ---")

    # 1. Розділяємо дані
    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Тренувальний набір: {train_data.count()} рядків")
    print(f"Тестовий набір: {test_data.count()} рядків")

    # 2. Визначаємо наші 3 моделі
    lr = LogisticRegression(featuresCol='features', labelCol='label')
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
    rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=20, maxDepth=5)

    models = [lr, dt, rf]
    model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]

    # 3. Визначаємо оцінювачі (Evaluators)
    # Ми перевіримо всі 4 метрики, як вимагає завдання [cite: 77]
    eval_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                       metricName="weightedPrecision")
    eval_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                    metricName="weightedRecall")

    print("\n--- Порівняння результатів (Завдання 6) ---")
    print("Модель | Accuracy | F1 Score | Precision | Recall")
    print("-" * 50)

    # 4. Навчаємо, прогнозуємо та оцінюємо кожну модель
    for model, name in zip(models, model_names):
        try:
            # Навчаємо модель
            print(f"Навчання {name}...")
            trained_model = model.fit(train_data)

            # Робимо прогнози
            predictions = trained_model.transform(test_data)

            # Оцінюємо якість
            accuracy = eval_accuracy.evaluate(predictions)
            f1 = eval_f1.evaluate(predictions)
            precision = eval_precision.evaluate(predictions)
            recall = eval_recall.evaluate(predictions)

            print(f"{name} | {accuracy:.2f} | {f1:.2f} | {precision:.2f} | {recall:.2f}")

        except Exception as e:
            print(f"Помилка під час навчання {name}: {e}")