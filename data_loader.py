from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType

def load_data(spark: SparkSession, file_path: str):
    """
    Завантажує дані NYC Taxi з CSV, використовуючи чітко визначену схему.
    """

    # Крок 1: Визначаємо схему на основі того, що ми знайшли
    # Це набагато ефективніше, ніж inferSchema=True
    schema = StructType([
        StructField("medallion", StringType(), True),
        StructField("hack_license", StringType(), True),
        StructField("vendor_id", StringType(), True),
        StructField("rate_code", IntegerType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("pickup_datetime", TimestampType(), True),
        StructField("dropoff_datetime", TimestampType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("trip_time_in_secs", IntegerType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True)
    ])

    # Крок 2: Зчитуємо CSV з нашою схемою
    df = spark.read.csv(
        file_path,
        header=True,
        schema=schema  # Застосовуємо нашу чітку схему
    )

    return df