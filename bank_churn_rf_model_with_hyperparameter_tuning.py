from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import CrossValidator , ParamGridBuilder
from pyspark.sql import  SparkSession
from pyspark.ml.feature import  VectorAssembler , StringIndexer
from pyspark.sql.types import  *
from pyspark.ml.classification  import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('datasets').getOrCreate()
df = spark.read.csv( '/content/drive/MyDrive/bank-churn-dataset/Churn_Modelling.csv', header = True, inferSchema = True)
df = df.dropna()
df = df.drop('surname')

train , test = df.randomSplit([0.8,0.2], seed = 42)

def data_clean(df): 
    label_col = 'Exited'
    ignore_cols = [label_col, 'RowNumber', 'CustomerId']  # Optional: include columns you want to exclude from features
    string_cols = [field.name for field in df.schema if isinstance(field.dataType, StringType) and field.name not in ignore_cols]
    output_cols = [col + '_indexer' for col in string_cols]

    indexer = StringIndexer(inputCols=string_cols, outputCols=output_cols)
    df_indexer = indexer.fit(df).transform(df).drop(*string_cols)

    feature_cols = [col for col in df_indexer.columns if col not in ignore_cols + [label_col]]
    vector = VectorAssembler(inputCols=feature_cols, outputCol='vector')
    df_vector = vector.transform(df_indexer)

    return 'vector', df_vector

col_name, train = data_clean(train)

rf  = RandomForestClassifier( featuresCol = col_name , labelCol = 'Exited', maxDepth = 5, maxBins = 24, numTrees = 21)

# pipeline = Pipeline(stages = [vector])
params = ( ParamGridBuilder()
    .addGrid(rf.maxDepth,[2,4,6])
    .addGrid(rf.maxBins, [20,30,40])
    .addGrid(rf.numTrees,[2,25,30])
    .build()
    )

cv = CrossValidator( estimator =  rf, estimatorParamMaps = params , evaluator = BinaryClassificationEvaluator(labelCol='Exited') , numFolds = 2 )
model = cv.fit(train)
_,test = data_clean(test)

prediction = model.transform(test)
evaluation = BinaryClassificationEvaluator( labelCol = 'Exited' , rawPredictionCol = 'rawPrediction', metricName = 'areaUnderPR')
auc_pr = evaluation.evaluate(prediction)


print(f' the area under pr : {auc_pr}')


'''testing ''' 

from pyspark.sql import Row

test_data_rows = [
    Row(RowNumber=10001, CustomerId=19000001, Geography="Germany", Gender="Male", CreditScore=580, Age=36, Tenure=4, Balance=121000.0, NumOfProducts=1, HasCrCard=1, IsActiveMember=0, EstimatedSalary=97000.0, Exited=1),
    Row(RowNumber=10002, CustomerId=19000002, Geography="Spain", Gender="Female", CreditScore=720, Age=28, Tenure=5, Balance=110000.0, NumOfProducts=2, HasCrCard=0, IsActiveMember=1, EstimatedSalary=42000.0, Exited=0),
    Row(RowNumber=10003, CustomerId=19000003, Geography="France", Gender="Female", CreditScore=670, Age=40, Tenure=7, Balance=143000.0, NumOfProducts=3, HasCrCard=1, IsActiveMember=1, EstimatedSalary=130000.0, Exited=0),
    Row(RowNumber=10004, CustomerId=19000004, Geography="Germany", Gender="Male", CreditScore=510, Age=50, Tenure=3, Balance=1000.0, NumOfProducts=1, HasCrCard=0, IsActiveMember=0, EstimatedSalary=39000.0, Exited=1),
    Row(RowNumber=10005, CustomerId=19000005, Geography="Spain", Gender="Male", CreditScore=600, Age=38, Tenure=6, Balance=98000.0, NumOfProducts=2, HasCrCard=1, IsActiveMember=1, EstimatedSalary=57000.0, Exited=0),

    Row(RowNumber=10006, CustomerId=19000006, Geography="France", Gender="Male", CreditScore=490, Age=43, Tenure=2, Balance=200000.0, NumOfProducts=4, HasCrCard=1, IsActiveMember=0, EstimatedSalary=85000.0, Exited=1),
    Row(RowNumber=10007, CustomerId=19000007, Geography="Germany", Gender="Female", CreditScore=750, Age=31, Tenure=9, Balance=89000.0, NumOfProducts=1, HasCrCard=1, IsActiveMember=1, EstimatedSalary=123000.0, Exited=0),
    Row(RowNumber=10008, CustomerId=19000008, Geography="France", Gender="Female", CreditScore=610, Age=45, Tenure=5, Balance=113000.0, NumOfProducts=2, HasCrCard=0, IsActiveMember=1, EstimatedSalary=76000.0, Exited=1),
    Row(RowNumber=10009, CustomerId=19000009, Geography="Spain", Gender="Male", CreditScore=590, Age=34, Tenure=6, Balance=124000.0, NumOfProducts=3, HasCrCard=0, IsActiveMember=1, EstimatedSalary=69000.0, Exited=0),
    Row(RowNumber=10010, CustomerId=19000010, Geography="France", Gender="Male", CreditScore=680, Age=41, Tenure=7, Balance=116000.0, NumOfProducts=2, HasCrCard=1, IsActiveMember=0, EstimatedSalary=45000.0, Exited=1)
]

test_df = spark.createDataFrame(test_data_rows)
_,test_df = data_clean(test_df)

prediction = model.transform(test_df)

evaluation = BinaryClassificationEvaluator( labelCol = 'Exited' , rawPredictionCol = 'rawPrediction', metricName = 'areaUnderPR')
auc_pr = evaluation.evaluate(prediction)


print(f' the roc : {auc_pr}')
