from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

# Запуск в терминале работает: spark-submit spark.py

# Файлы данных
file_crime = 'data/crime.csv'
file_offense_codes = 'data/offense_codes.csv'
# Путь к каталогу - где сохраняется сформированные данные в формате parquet
path_save_folder = 'output/showcase_crime'

# Запустить сессию spark
spark = (SparkSession.builder
         .master("local[*]")
         .appName('pyspark_mrm')
         .getOrCreate())

# Получить данные файла crime.csv
df_crime = (spark.read.format("csv")
            .option("mode", "FAILFAST")
            .option("inferSchema", "true")
            .option("header", "true")
            .option("encoding", "latin1")
            .option("path", file_crime)
            .load())

# Получить данные файла offense_codes.csv
df_offense_codes = (spark.read.format("csv")
                    .option("mode", "FAILFAST")
                    .option("inferSchema", "true")
                    .option("header", "true")
                    .option("encoding", "latin1")
                    .option("path", file_offense_codes)
                    .load())

# Получить только необходимые данные для нашей задачи
df_crime = df_crime.select("DISTRICT", "YEAR", "MONTH", "OFFENSE_CODE", "Lat", "Long")

# Получить crime_type - это первая часть NAME из таблицы offense_codes, разбитого по разделителю “-”
# и удалить дубликаты уникальных кодов.
df_crime_type = (df_offense_codes
                 .withColumn("row_num", row_number().over(Window.partitionBy("CODE").orderBy(col("NAME").desc())))
                 .filter(col("row_num") == 1)
                 .drop("row_num")
                 .withColumn('CRIME_TYPE', substring_index("NAME", " -", 1))
                 .orderBy(col("CODE")))

# Добавить к основным данным df_crime, полученный новый crime_type (только этот столбец, другие удаляются)
df_crime = (df_crime.join(broadcast(df_crime_type), df_crime['OFFENSE_CODE'] == df_crime_type['CODE'])
            .drop("CODE", "NAME")
            .cache())

# Получить общее количество преступлений в районе
df_crimes_total = df_crime.groupBy('DISTRICT').count().select('DISTRICT', col('count').alias('crimes_total'))

# Получить медиану числа преступлений в месяц в районе
df_crimes_monthly = (df_crime.groupBy("DISTRICT", "YEAR", "MONTH")
                     .agg(count("*").alias("month_crimes"))
                     .groupBy("DISTRICT")
                     .agg(expr('percentile_approx(month_crimes, 0.5)').alias("crimes_monthly")))

#  Получить три самых частых crime_type за всю историю наблюдений в каждом районе
df_frequent_crime_types = (df_crime.groupBy("DISTRICT", "CRIME_TYPE")
                           .agg(count("*").alias("crimes_by_crime_type"))
                           .withColumn("row_num", row_number().over(Window.partitionBy("DISTRICT")
                                                                    .orderBy(col("crimes_by_crime_type").desc())))
                           .filter(col("row_num") <= 3).drop("row_num", "crimes_by_crime_type")
                           .groupBy("DISTRICT")
                           .agg(concat_ws(", ", collect_list(col("CRIME_TYPE"))).alias("frequent_crime_types")))

# Были нулевые/пустые значения и странные координаты (ровно -1) в широтах и долготах.
# Я их убрал для корректного расчёта, надеюсь правильно сделал.
# Получить среднюю широту координат инцидентов в районе
df_lat = df_crime.where("Lat<>-1 AND Lat IS NOT NULL").groupBy(col("DISTRICT")).agg(avg(col("Lat")).alias("lat"))
# Получить среднюю долготу координат инцидентов в районе
df_lng = df_crime.where("Long<>-1 AND Long IS NOT NULL").groupBy(col("DISTRICT")).agg(avg(col("Long")).alias("lng"))

# Обьединить данные по районам
df_result = (df_crimes_total.join(df_crimes_monthly, ['DISTRICT'])
             .join(df_frequent_crime_types, ['DISTRICT'])
             .join(df_lat, ['DISTRICT'])
             .join(df_lng, ['DISTRICT'])
             .select("DISTRICT", "crimes_total", "crimes_monthly", "frequent_crime_types", "lat", "lng")
             .orderBy(col("DISTRICT")))
# df_result.show()

# Сохранить итоговый набор данных df_result в файл, формате parquet
df_result.coalesce(1).write.mode('overwrite').parquet(path_save_folder)
# df_result.write.save(path_save_folder, format='parquet')

# Остановить сессию spark
spark.stop()

# Итоговые данные:
# +--------+------------+--------------+--------------------+------------------+------------------+
# |DISTRICT|crimes_total|crimes_monthly|frequent_crime_types|               lat|               lng|
# +--------+------------+--------------+--------------------+------------------+------------------+
# |      A1|       35717|           904|PROPERTY, ASSAULT...|42.356733944169925|-71.06112996712561|
# |     A15|        6505|           160|M/V ACCIDENT, INV...| 42.37599933215882|-71.06267564355912|
# |      A7|       13544|           344|SICK/INJURED/MEDI...| 42.37735545951605|-71.03083363200777|
# |      B2|       49945|          1298|M/V, M/V ACCIDENT...| 42.32162901059059|-71.08479984973216|
# |      B3|       35442|           907|VERBAL DISPUTE, I...| 42.28691276523002|-71.08518799423317|
# |     C11|       42530|          1115|M/V, SICK/INJURED...| 42.30005270854692|-71.06325855819784|
# |      C6|       23460|           593|DRUGS, SICK/INJUR...|42.333628045765124|-71.05203273306276|
# |     D14|       20127|           505|TOWED MOTOR VEHIC...| 42.35030978352097| -71.1422613535374|
# |      D4|       41915|          1084|LARCENY SHOPLIFTI...| 42.34350022705704|-71.08090067663328|
# |     E13|       17536|           445|SICK/INJURED/MEDI...|42.315022018702614|-71.10645084032734|
# |     E18|       17348|           435|SICK/INJURED/MEDI...|  42.2626806112259|-71.11891998757716|
# |      E5|       13239|           337|SICK/INJURED/MEDI...| 42.28247306566417|-71.14134995577206|
# +--------+------------+--------------+--------------------+------------------+------------------+

