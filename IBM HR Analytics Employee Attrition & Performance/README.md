# 직원 이직 예측 분석 (Employee Attrition Prediction)

IBM HR Analytics 데이터셋을 기반으로, 직원의 이직 여부를 예측하는 모델을 구축한 분석 프로젝트

---

## 데이터셋 정보

- 출처: [Kaggle - IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 관측치 수: 1,470건
- 타겟 컬럼: `Attrition` (Yes, No)
- 

---

## 프로젝트 진행 과정

### 1. 데이터 로딩 및 불필요한 컬럼 제거

```python
drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
hr_df = df.drop(*drop_cols)
```

- 'EmployeeCount', 'StandardHours': 모든 값이 동일하므로 모델 학습에 도움이 되지 않음
- 'Over18': 모든 값이 'Y'로 동일 → 정보 없음
- 'EmployeeNumber': 고유 식별자 역할 → 예측에 영향을 주지 않음

---

### 2. 결측치 확인

```python
hr_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in hr_df.columns]).show()
```

- 모든 컬럼에 결측치 없음

---

### 3. 타겟 변수 라벨 인코딩

```python
label_indexer = StringIndexer(inputCol='Attrition', outputCol='label')
```

- `Attrition` 컬럼을 0, 1로 변환 (No → 0, Yes → 1)

---

### 4. 범주형 변수 처리

```python
cat_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
```

각 범주형 변수에 대해 다음과 같은 처리를 진행:
- `StringIndexer`: 범주형 → 숫자형 인코딩
- `OneHotEncoder`: 희소 벡터로 변환하여 모델에 적합하게 구성

---

### 5. 수치형 변수 정규화

```python
num_features = [...]
```

- `VectorAssembler`로 벡터화 후 `StandardScaler`로 표준 정규화 처리

---

### 6. 피처 벡터 통합

```python
assembler_input = [cat+'_onehot' for cat in cat_features] + [num+'_scaled' for num in num_features]
assembler = VectorAssembler(inputCols=assembler_input, outputCol='features')
```

---

### 7. 머신러닝 파이프라인 구성 및 적용

```python
stages = [label_indexer] + cat_stages + num_vector_stages + [assembler]
pipeline = Pipeline(stages=stages)
```

- 파이프라인을 통해 전체 전처리 과정 자동화

---

## 모델 학습 및 평가

### 데이터 분할 및 파이프라인 학습

```python
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
fitted_pipeline = pipeline.fit(train_df)
vtrain_df = fitted_pipeline.transform(train_df)
vtest_df = fitted_pipeline.transform(test_df)
```

---

### 로지스틱 회귀 모델 학습

```python
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(vtrain_df)
```

---

### 예측 결과 및 평가

```python
pred = model.transform(vtest_df)
pred.select('label', 'prediction', 'probability').show()
```

- 평가 지표: AUC (Area Under ROC Curve)

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(pred)
```

---

