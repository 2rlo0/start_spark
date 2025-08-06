from pyspark.sql import SparkSession
spark1 = SparkSession.builder.appName("reload mnm data").getOrCreate()
output_path = 'output/mnms_result'

result_rdd = spark1.sparkContext.textFile(output_path)
print('결과 미리보기-----')

for line in result_rdd.take(5):
    print(line)

print(f'총 데이터 라인 수: {result_rdd.count()}')

spark1.stop()