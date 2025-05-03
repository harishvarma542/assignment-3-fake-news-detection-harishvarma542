from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNewsClassifier").getOrCreate()

# --------------------------
# Task 1: Load & Basic Exploration
# --------------------------
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

df.show(5)
print("Total Articles:", df.count())
df.select("label").distinct().show()

df.write.csv("output/task1_output.csv", header=True)

# --------------------------
# Task 2: Text Preprocessing
# --------------------------
df_cleaned = df.withColumn("text", lower(col("text")))

tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized = tokenizer.transform(df_cleaned)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized).select("id", "title", "filtered_words", "label")

# Convert array column to string for CSV output
output_df = cleaned_df.withColumn("filtered_words_str", concat_ws(",", "filtered_words")).drop("filtered_words")
output_df.write.csv("output/task2_output.csv", header=True)

# --------------------------
# Task 3: Feature Extraction
# --------------------------
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized = hashingTF.transform(cleaned_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurized)
rescaled = idfModel.transform(featurized)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
final_data = indexer.fit(rescaled).transform(rescaled).select("id", "filtered_words", "features", "label_index")

# Convert filtered_words to string
final_data = final_data.withColumn("filtered_words_str", concat_ws(",", "filtered_words"))

# Convert TF-IDF vector to string
vector_to_string_udf = udf(lambda v: ",".join([f"{x:.6f}" for x in v.toArray()]), returnType="string")
final_data = final_data.withColumn("features_str", vector_to_string_udf("features"))

# Drop complex types
final_output_df = final_data.select("id", "filtered_words_str", "features_str", "label_index")

# Write to CSV
final_output_df.write.csv("output/task3_output.csv", header=True)


# --------------------------
# Task 4: Model Training
# --------------------------
train, test = final_data.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train)
predictions = model.transform(test)

predictions.select("id", "label_index", "prediction").write.csv("output/task4_output.csv", header=True)

# --------------------------
# Task 5: Evaluate the Model
# --------------------------
evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

# Save to CSV
spark.createDataFrame([
    ("Accuracy", accuracy),
    ("F1 Score", f1)
], ["Metric", "Value"]).write.csv("output/task5_output.csv", header=True)
