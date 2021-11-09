############################
# 1. Libraries
############################
import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


############################
# 2. Spark Session
############################
findspark.init(r"C:\spark")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_hw") \
    .getOrCreate()


sct = spark.sparkContext

############################
# 3. Exploratory Data Analysis
############################

spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)
spark_df
type(spark_df)


# Gözlem ve değişken sayısı
print("Shape: ", (spark_df.count(), len(spark_df.columns)))
# Shape:  (10000, 14)

# Değişken tipleri
spark_df.printSchema()
# root
# |-- RowNumber: integer (nullable = true)
# |-- CustomerId: integer (nullable = true)
# |-- Surname: string (nullable = true)
# |-- CreditScore: integer (nullable = true)
# |-- Geography: string (nullable = true)
# |-- Gender: string (nullable = true)
# |-- Age: integer (nullable = true)
# |-- Tenure: integer (nullable = true)
# |-- Balance: double (nullable = true)
# |-- NumOfProducts: integer (nullable = true)
# |-- HasCrCard: integer (nullable = true)
# |-- IsActiveMember: integer (nullable = true)
# |-- EstimatedSalary: double (nullable = true)
# |-- Exited: integer (nullable = true)



spark_df.show(5)
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# |RowNumber|CustomerId| Surname|CreditScore|Geography|Gender|Age|Tenure|  Balance|NumOfProducts|HasCrCard|IsActiveMember|EstimatedSalary|Exited|
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# |        1|  15634602|Hargrave|        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|
# |        2|  15647311|    Hill|        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|
# |        3|  15619304|    Onio|        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|
# |        4|  15701354|    Boni|        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|
# |        5|  15737888|Mitchell|        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# only showing top 5 rows



# Değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# |rownumber|customerid| surname|creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# |        1|  15634602|Hargrave|        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|
# |        2|  15647311|    Hill|        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|
# |        3|  15619304|    Onio|        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|
# |        4|  15701354|    Boni|        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|
# |        5|  15737888|Mitchell|        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|
# +---------+----------+--------+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+
# only showing top 5 rows



# özet istatistikler
spark_df.describe().show()
# |summary|         rownumber|       customerid|surname|      creditscore|geography|gender|               age|            tenure|          balance|     numofproducts|          hascrcard|     isactivemember|  estimatedsalary|             exited|
# +-------+------------------+-----------------+-------+-----------------+---------+------+------------------+------------------+-----------------+------------------+-------------------+-------------------+-----------------+-------------------+
# |  count|             10000|            10000|  10000|            10000|    10000| 10000|             10000|             10000|            10000|             10000|              10000|              10000|            10000|              10000|
# |   mean|            5000.5|  1.56909405694E7|   null|         650.5288|     null|  null|           38.9218|            5.0128|76485.88928799961|            1.5302|             0.7055|             0.5151|100090.2398809998|             0.2037|
# | stddev|2886.8956799071675|71936.18612274907|   null|96.65329873613035|     null|  null|10.487806451704587|2.8921743770496837|62397.40520238599|0.5816543579989917|0.45584046447513327|0.49979692845891815|57510.49281769821|0.40276858399486065|
# |    min|                 1|         15565701|  Abazu|              350|   France|Female|                18|                 0|              0.0|                 1|                  0|                  0|            11.58|                  0|
# |    max|             10000|         15815690| Zuyeva|              850|    Spain|  Male|                92|                10|        250898.09|                 4|                  1|                  1|        199992.48|                  1|
# +-------+------------------+-----------------+-------+-----------------+---------+------+------------------+------------------+-----------------+------------------+-------------------+-------------------+-----------------+-------------------+

# sadece belirli değişkenler için özet istatistikler
spark_df.describe(["age", "exited"]).show()


# Kategorik değişken sınıf istatistikleri
spark_df.groupby("exited").count().show()
# |exited|count|
# +------+-----+
# |     1| 2037|
# |     0| 7963|
# +------+-----+


# eşsiz sınıflar
spark_df.select("exited").distinct().show()

# groupby transactions
spark_df.groupby("exited").count().show()
spark_df.groupby("exited").agg({"tenure": "mean"}).show()


#  Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()


# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']


# hedef değişkene göre sayısal değişkenlerin ortalaması
for col in [col.lower() for col in num_cols]:
    spark_df.groupby("exited").agg({col: "mean"}).show()



##################################################
# 4. Data Preprocessing & Feature Engineering
##################################################

############################
## 4.1  Missing Values
############################

from pyspark.sql.functions import when, count, col
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T
# rownumber        0
# customerid       0
# surname          0
# creditscore      0
# geography        0
# gender           0
# age              0
# tenure           0
# balance          0
# numofproducts    0
# hascrcard        0
# isactivemember   0
# estimatedsalary  0
# exited           0




# spark tarafındaki nesneyi pandas dünyasına getiriyoruz. böylelikle pandas
# methodlarını uygulayabiliyoruz


############################
## 4.2 Feature Interaction
############################
# gerekli olmayan kolonların silinmesi
spark_df = spark_df.drop('rownumber', "customerid", "surname")


spark_df = spark_df.withColumn('creditscore_salary', spark_df.creditscore / spark_df.estimatedsalary)
spark_df = spark_df.withColumn('creditscore_tenure', spark_df.creditscore * spark_df.tenure)
spark_df = spark_df.withColumn('balance_salary', spark_df.balance / spark_df.estimatedsalary)
spark_df.show(5)

# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+
# |creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+
# |        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|
# |        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|
# |        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|
# |        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|
# |        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+
# only showing top 5 rows

############################
## 4.3 Bucketization / Bining / Num to Cat
############################

# age değişkeni
spark_df.select('age').describe().toPandas().transpose()
spark_df.select("age").summary("count", "min", "25%", "50%","75%", "max").show()
bucketizer = Bucketizer(splits=[0, 35, 55, 75, 95], inputCol="age", outputCol="age_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)
# |summary|  age|
# +-------+-----+
# |  count|10000|
# |    min|   18|
# |    25%|   32|
# |    50%|   37|
# |    75%|   44|
# |    max|   92|
# +-------+-----+#


spark_df.groupby("age_cat").count().show()
spark_df.groupby("age_cat").agg({'exited': "mean"}).show()

# float değerleri intere çevirmek
spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))


# WHEN KULLANIMI
spark_df.withColumn('creditscore_2',
                    when(spark_df['creditscore'] < 301, "deep").
                    when((301 < spark_df['creditscore']) & (spark_df['creditscore'] < 601), "very poor").
                    when((500 < spark_df['creditscore']) & (spark_df['creditscore'] < 601), "poor").
                    when((601 < spark_df['creditscore']) & (spark_df['creditscore'] < 661), "fair").
                    when((661 < spark_df['creditscore']) & (spark_df['creditscore'] < 781), "good").
                    when((781 < spark_df['creditscore']) & (spark_df['creditscore'] < 851), "excellent").
                    otherwise("top")).show()



+-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+
|creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|creditscore_2|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+
# |        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|         fair|
# |        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|         fair|
# |        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|    very poor|
# |        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|         good|
# |        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|    excellent|
# |        645|    Spain|  Male| 44|     8|113755.78|            2|        1|             0|      149756.71|     1|0.004306985643581513|              5160|0.7596038935417319|      2|         fair|
# |        822|   France|  Male| 50|     7|      0.0|            2|        1|             1|        10062.8|     0| 0.08168700560480185|              5754|               0.0|      2|    excellent|
# |        376|  Germany|Female| 29|     4|115046.74|            4|        1|             0|      119346.88|     1|0.003150480347705...|              1504|0.9639693974404694|      1|    very poor|
# |        501|   France|  Male| 44|     4|142051.07|            2|        0|             1|        74940.5|     0|0.006685303674248237|              2004|1.8955180443151567|      2|    very poor|
# |        684|   France|  Male| 27|     2|134603.88|            1|        1|             1|       71725.73|     0|0.009536326782592523|              1368| 1.876647055387237|      1|         good|
# |        528|   France|  Male| 31|     6|102016.72|            2|        0|             0|       80181.12|     0|0.006585091353176409|              3168|1.2723284483928388|      1|    very poor|
# |        497|    Spain|  Male| 24|     3|      0.0|            2|        1|             0|       76390.01|     0|0.006506086332492954|              1491|               0.0|      1|    very poor|
# |        476|   France|Female| 34|    10|      0.0|            2|        1|             0|       26260.98|     0|0.018125751590382384|              4760|               0.0|      1|    very poor|
# |        549|   France|Female| 25|     5|      0.0|            2|        0|             0|      190857.79|     0|0.002876487252629...|              2745|               0.0|      1|    very poor|
# |        635|    Spain|Female| 35|     7|      0.0|            2|        1|             1|       65951.65|     0|0.009628265555145323|              4445|               0.0|      2|         fair|
# |        616|  Germany|  Male| 45|     3|143129.41|            2|        0|             1|       64327.26|     0|0.009576033550939368|              1848|2.2250195329320728|      2|         fair|
# |        653|  Germany|  Male| 58|     1|132602.88|            1|        1|             0|        5097.67|     1|  0.1280977387708502|               653| 26.01244882465911|      3|         fair|
# |        549|    Spain|Female| 24|     9|      0.0|            2|        1|             1|       14406.41|     0| 0.03810803663091638|              4941|               0.0|      1|    very poor|
# |        587|    Spain|  Male| 45|     6|      0.0|            1|        0|             0|      158684.81|     0|0.003699156838011...|              3522|               0.0|      2|    very poor|
# |        726|   France|Female| 24|     6|      0.0|            2|        1|             1|       54724.03|     0|0.013266566808036616|              4356|               0.0|      1|         good|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+
# only showing top 20 rows




##################################################
## 4.4 User Defined Functions (UDFs)
##################################################
from pyspark.sql.types import IntegerType, StringType, FloatType
from pyspark.sql.functions import udf

# udf ile fonksiyon yazma
def segment(tenure):
    if tenure < 5:
        return "segment_b"
    else:
        return "segment_a"

func_udf = udf(segment, StringType())
spark_df = spark_df.withColumn('segment', func_udf(spark_df['tenure']))
spark_df.show(5)
|creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|  segment|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+---------+
# |        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|segment_b|
# |        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|segment_b|
# |        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|segment_a|
# |        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|segment_b|
# |        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|segment_b|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+---------+
# only showing top 5 rows#


spark_df.groupby("segment").count().show()
# |  segment|count|
# +---------+-----+
# |segment_a| 5506|
# |segment_b| 4494|
# +---------+-----+


############################
# 4.5 Label Encoding
############################

spark_df.show(5)

indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))
spark_df = spark_df.drop('segment')
# |creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|  segment|segment_label|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+---------+-------------+
# |        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|segment_b|          1.0|
# |        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|segment_b|          1.0|
# |        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|segment_a|          0.0|
# |        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|segment_b|          1.0|
# |        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|segment_b|          1.0|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+---------+-------------+
only showing top 5 rows



spark_df.show(5)
# |creditscore|geography|gender|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|segment_label|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+
# |        619|   France|Female| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|            1|
# |        608|    Spain|Female| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|            1|
# |        502|   France|Female| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|            0|
# |        699|   France|Female| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|            1|
# |        850|    Spain|Female| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|            1|
# +-----------+---------+------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+
# only showing top 5 rows



indexer = StringIndexer(inputCol="gender", outputCol="gender_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_label", temp_sdf["gender_label"].cast("integer"))
spark_df = spark_df.drop('gender')

spark_df.show(5)
# |creditscore|geography|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|segment_label|gender_label|
# +-----------+---------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+------------+
# |        619|   France| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|            1|           1|
# |        608|    Spain| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|            1|           1|
# |        502|   France| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|            0|           1|
# |        699|   France| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|            1|           1|
# |        850|    Spain| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|            1|           1|
# +-----------+---------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+------------+
# only showing top 5 rows




indexer = StringIndexer(inputCol="geography", outputCol="geography_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("geography_label", temp_sdf["geography_label"].cast("integer"))
spark_df = spark_df.drop('geography')

spark_df.show(5)
# |creditscore|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|segment_label|gender_label|geography_label|
# +-----------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+------------+---------------+
# |        619| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|            1|           1|              0|
# |        608| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|            1|           1|              2|
# |        502| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|            0|           1|              0|
# |        699| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|            1|           1|              0|
# |        850| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|            1|           1|              2|
# +-----------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+------------+---------------+
# only showing top 5 rows



############################
# 4.6 One Hot Encoding
############################
spark_df.show(5)

encoder = OneHotEncoder(inputCols=["age_cat", "geography_label"], outputCols=["age_cat_ohe", "geography_label_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)



############################
# 4.7 Defining TARGET
############################

stringIndexer = StringIndexer(inputCol='exited', outputCol='label')

temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
temp_sdf.show()
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)
# |creditscore|age|tenure|  balance|numofproducts|hascrcard|isactivemember|estimatedsalary|exited|  creditscore_salary|creditscore_tenure|    balance_salary|age_cat|segment_label|gender_label|geography_label|  age_cat_ohe|geography_label_ohe|label|
# +-----------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+------------+---------------+-------------+-------------------+-----+
# |        619| 42|     2|      0.0|            1|        1|             1|      101348.88|     1|0.006107615594765329|              1238|               0.0|      2|            1|           1|              0|(4,[2],[1.0])|      (2,[0],[1.0])|    1|
# |        608| 41|     1| 83807.86|            1|        0|             1|      112542.58|     0|0.005402399696186101|               608|0.7446769036217226|      2|            1|           1|              2|(4,[2],[1.0])|          (2,[],[])|    0|
# |        502| 42|     8| 159660.8|            3|        1|             0|      113931.57|     1|0.004406153623618106|              4016|1.4013745268322026|      2|            0|           1|              0|(4,[2],[1.0])|      (2,[0],[1.0])|    1|
# |        699| 39|     1|      0.0|            2|        0|             0|       93826.63|     0|0.007449910542454738|               699|               0.0|      2|            1|           1|              0|(4,[2],[1.0])|      (2,[0],[1.0])|    0|
# |        850| 43|     2|125510.82|            1|        1|             1|        79084.1|     0|0.010748051757559357|              1700|1.5870550464631954|      2|            1|           1|              2|(4,[2],[1.0])|          (2,[],[])|    0|
# +-----------+---+------+---------+-------------+---------+--------------+---------------+------+--------------------+------------------+------------------+-------+-------------+------------+---------------+-------------+-------------------+-----+
# only showing top 5 rows




############################
# 4.8 Defining Features
############################

cols = ['creditscore', 'age', 'tenure', 'balance','numofproducts', 'hascrcard',
        'isactivemember', 'estimatedsalary', 'creditscore_salary', 'creditscore_tenure',
        'balance_salary', 'segment_label', 'gender_label',
        'age_cat_ohe', 'geography_label_ohe']



# Vectorize independent variables.
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()



# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)
# |            features|label|
# +--------------------+-----+
# |[619.0,42.0,2.0,0...|    1|
# |[608.0,41.0,1.0,8...|    0|
# |[502.0,42.0,8.0,1...|    1|
# |(19,[0,1,2,4,7,8,...|    0|
# |[850.0,43.0,2.0,1...|    0|
# +--------------------+-----+
# only showing top 5 rows

# StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
final_df = scaler.fit(final_df).transform(final_df)



# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(5)
# |            features|label|     scaled_features|
# +--------------------+-----+--------------------+
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|
# |(19,[0,1,2,3,4,5,...|    1|(19,[0,1,2,3,4,5,...|
# +--------------------+-----+--------------------+
# only showing top 5 rows#


test_df.show(5)


print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

# Training Dataset Count: 6949
# Test Dataset Count: 3051


##################################################
# 5. Modeling
##################################################

############################
# 5.1 Logistic Regression
############################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
# 0.8174369059324812


evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))
# accuracy: 0.817437, precision: 0.832153, recall: 0.965886, f1: 0.781988, roc_auc: 0.599448

############################
# 5.2 Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)
# |            features|label|     scaled_features|       rawPrediction|         probability|prediction|
# +--------------------+-----+--------------------+--------------------+--------------------+----------+
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|[1.75024081719644...|[0.97070147009728...|       0.0|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|[1.26360126567415...|[0.92602695213284...|       0.0|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|[1.21202822933281...|[0.91864343014702...|       0.0|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|[1.75728443667020...|[0.97109946729348...|       0.0|
# |(19,[0,1,2,3,4,5,...|    0|(19,[0,1,2,3,4,5,...|[1.14318764430859...|[0.90774234263874...|       0.0|
# +--------------------+-----+--------------------+--------------------+--------------------+----------+
# only showing top 5 rows#


y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
#   0.8597181252048509


############################
# 5.3 Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)
y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# 0.866601114388725