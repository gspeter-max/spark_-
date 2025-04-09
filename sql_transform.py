
from   pyspark.sql import SparkSession
from pyspark.ml.feature import SQLTransformer

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Sample data
data = [
    ("A", 25, 50000, 2),
    ("B", 30, 60000, 4)
]

# Create DataFrame
columns = ["Name", "Age", "Salary", "Experience"]
df = spark.createDataFrame(data, columns)

sqltrans = SQLTransformer(statement = 'select * , Salary / experience  as SalaryPerYear , 65 - Age as RetireIn from __THIS__')
sqltrans.transform(df).show()

sqltrans = SQLTransformer(statement = '''select * ,
        Salary / experience  as SalaryPerYear,
            65 - Age as RetireIn,
            case when salary <= 55000 then "Low" when salary between 55000 and 65000 then "Medium" else "High" end as incomeBrakit,
            case when Experience <= 2 then 1 else 0 end as isJunior ,
            Round( Salary / experience , 0) as RoundedSalaryPerYear

        from __THIS__''')

sqltrans.transform(df).show()

from   pyspark.sql import SparkSession
from pyspark.ml.feature import SQLTransformer
from pyspark.sql.window import Window
import  pyspark.sql.functions as f
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Sample data
data = [
    ("A", 25, 50000, 2),
    ("B", 30, 60000, 4)
]

# Create DataFrame
columns = ["Name", "Age", "Salary", "Experience"]
df = spark.createDataFrame(data, columns)
# spark_row = SQLTransformer(statement = 'select * , row_number() over(order by Index ) as rank from __THIS__')

sqltrans = SQLTransformer(statement = '''select * ,
        Salary / experience  as SalaryPerYear,
            65 - Age as RetireIn,
            case when salary <= 55000 then "Low" when salary between 55000 and 65000 then "Medium" else "High" end as incomeBrakit,
            case when Experience <= 2 then 1 else 0 end as isJunior ,
            Round( Salary / experience , 0) as RoundedSalaryPerYear,
            salary + Rand() * 5000 as future_salary,
            case when age < 28 then "Young" when age between 28 and 35 then "Mid" else "Senior" end as Age_Group,
            Experience * 2 + (65 - Age) * 0.5 as career_Score,
            row_number() over( order by Name ) as rank

        from __THIS__
        ''')

spark2 = SQLTransformer(statement = '''
        select *,Name , Age , Salary, Experience, rank
        from __THIS__
        union all

        select *,Name, Age , Salary + 10000 ,Experience , rank + 1000
        from __THIS__
        where Experience  = 2
''')
df2 = sqltrans.transform(df)
df3 = spark2.transform(df2)
spark3 = SQLTransformer(statement = '''
    select * ,
        case when rank > 1000 then 1 else 0 end is_duplicates,
        career_Score + (future_salary / 10000) as adjustedScore,
        case when career_Score + (future_salary / 10000) > 70  then "HighPotential" when career_Score + (future_salary / 10000) between 60 and 70 then "MidPotential" else "LowPotential" end as Potentail
    from __THIS__

'''
)
spark3.transform(df3).show()

spark4 = SQLTransformer(
    statement = '''
        select * ,
            sum( case when isJunior = 1 then 1 else 0 end ) over ()  as gpt_was_not_give_me_thatname

        from __THIS__

'''
)
df4 = spark4.transform(df3)

spark5 = SQLTransformer(
    statement = '''
        select * ,
            Name,
            Salary,
            rank
        from __THIS__


        union all

        select *,
            Concat(Name, '_dup1') as Name,
            Salary + 5000,
            rank + 100
        from __THIS__

        union all

        select *,
            Concat(Name, '_dup2') as Name,
            Salary + 10000,
            rank + 200
        from __THIS__

        union all

        select *,
            Concat(Name, '_dup3') as Name,
            Salary + 15000,
            rank + 300
        from __THIS__

    '''
)
df5 = spark5.transform(df4)
