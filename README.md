# Description 


# Taks
1. Ejecute todas las celdas y obtenga los resultados

2. Proporcione los resultados de la tabla classification_report debajo de esta tarea para el modelo LogisticRegression.

3. Aplique 2 algoritmos alternativos al utilizado para resolver el problema de clasificación (por ejemplo, XGBClassifier y algún otro más) y obtenga los resultados en la tabla classification_report.

4. Para XGBClassifier necesitará definir los parámetros:
learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27

5. En la sección de vectorización TF-IDF, de manera análoga a los unigramas y pentagramas, calcule el classification_report para bigramas y trigramas. Publique los resultados en el informe e indique si cambió la precisión (f1-score) al usarlos en comparación con los unigramas y pentagramas.


# Самостоятельная работа

1. Изучите материал, представленный в борде.
2. Выполните все ячейки и получите результаты.
3. Приведите результаты таблицы classification_report в под этим заданием для модели LogisticRegression
4. Примените 2 альтернативных использованному алгоритму для решения задачи классификации (для примера XGBClassifier и еще какой-то один) и получите результаты в таблице classification_report
5. Для XGBClassifier вам потребуется задать параметры
```learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27```

6. В разделе TF-IDF векторизация по аналогии с униграммами и пентаграммами вычислите classification_report для биграмм, триграмм опубликуйте результаты в отчете и укажите изменилась ли точность f1-score при их использовании по сравнению с униграммами и пентаграммами.


**Pasos en el proceso:** 

1. Cargar los CSV en pandas y combinar/etiquetar.
2. Preprocesar texto (limpieza básica).
3. Vectorizar texto (por ejemplo, TF-IDF).
4. Entrenar un clasificador LogisticRegression.
5. Predecir sobre conjunto de prueba.
6. Mostrar classification_report (precision, recall, f1-score y support).

=======================================
  RESULTATS - COMPARISON OF ALL MODELS
=======================================

 TF-ID УНИГРАМ - LogistRegresion ================
              precision    recall  f1-score   support

    negative       0.72      0.46      0.56     28108
    positive       0.61      0.82      0.70     28601

    accuracy                           0.64     56709
   macro avg       0.66      0.64      0.63     56709
weighted avg       0.66      0.64      0.63     56709

=============
 TF-ID ВЫНИГРАМ - LogistRegresion ================
              precision    recall  f1-score   support

    negative       0.72      0.66      0.69     28108
    positive       0.69      0.75      0.72     28601

    accuracy                           0.71     56709
   macro avg       0.71      0.71      0.71     56709
weighted avg       0.71      0.71      0.71     56709

=============
 TF-ID Триграмма - LogistRegresion ================
              precision    recall  f1-score   support

    negative       0.73      0.45      0.56     28108
    positive       0.61      0.83      0.70     28601

    accuracy                           0.64     56709
   macro avg       0.67      0.64      0.63     56709
weighted avg       0.67      0.64      0.63     56709

=============
 TF-ID 5 Gram - LogistRegresion ================
Pentagramas:               precision    recall  f1-score   support

    negative       0.52      0.99      0.68     28108
    positive       0.95      0.11      0.19     28601

    accuracy                           0.55     56709
   macro avg       0.74      0.55      0.44     56709
weighted avg       0.74      0.55      0.44     56709

=============
  RESULTADOS XGBClassifier ================
              precision    recall  f1-score   support

    negative       0.75      0.68      0.71     28108
    positive       0.71      0.77      0.74     28601

    accuracy                           0.73     56709
   macro avg       0.73      0.73      0.73     56709
weighted avg       0.73      0.73      0.73     56709


=== KNN / K-ближайших соседей ===
              precision    recall  f1-score   support

    negative     0.5389    0.9719    0.6934     28108
    positive     0.8686    0.1829    0.3022     28601

    accuracy                         0.5739     56709
   macro avg     0.7038    0.5774    0.4978     56709
weighted avg     0.7052    0.5739    0.4961     56709


# Результаты анализа данных по отелю

Model                     Accuracy   F1-Score  
---------------------------------------------
Baseline (Unigrams)       0.4951     0.4949
TF-IDF (Unigrams)         0.4939     0.4934
TF-IDF (Trigrams)         0.5177     0.5130
XGBoost                   0.5020     0.5014
Random Forest             0.4846     0.4777


# Conclusiont of both Results

**For the Positive/Negative Sentiment Dataset:**
The models demonstrate acceptable performance (73% accuracy with XGBoost), indicating that the features (text data) contain discernible sentiment patterns.

**Для набора данных с положительной/отрицательной тональностью:**
Модели демонстрируют приемлемую производительность (точность 73% с XGBoost), что указывает на то, что признаки (текстовые данные) содержат различимые паттерны тональности.

**For the Hotel Review Dataset:**
The results are critically poor (~50% accuracy), essentially no better than random guessing. This strongly suggests fundamental issues with the data itself, such as poor labeling, highly ambiguous text, class imbalance, or that the sentiment signal is too weak or complex for the features extracted.

**Для набора данных с отзывами об отелях:**
Результаты критически низкие (~50% точности), по сути не лучше случайного угадывания. Это явно указывает на фундаментальные проблемы с самими данными: плохая разметка, сильно неоднозначный текст, дисбаланс классов или слишком слабый/сложный для извлеченных признаков сигнал тональности. 





# ====== Notes  - Констра ======:
**Concepts**

What are n-grams:
The smallest structures of the language we work with are called n-grams. The n-gram has a parameter n, which is the number of words that fall into this representation of the text.

If n = 1, then we look at how many times each word appeared in the text. We get unigrams
If n = 2, then we look at how many times each pair of consecutive words appeared in the text. We get bigrams

step and step


# Firts Get data Sample 
1. firts we get the data sample positive and negative
Сначала мы получаем образец данных
2. We organize the data, indicate that the table seperator is (;), and select column 3, which is the information we going to evaluate.
Мы организуем данные, указываем, что разделитеь таблицы - это (;) и выбираем столбец 3, который седержит информатцию для оценки
# Seconds fragmentation of the text

3. We use the module ntlk. It´s nGram library. Extract contiguous sequences of n elements (usually words or lettes) from a text.
мы используем модуль ntlk, это библиотека nGram. Извлекает непрерывные последовательности из n элементов (обычно слов или букв) из текста.

# Vectorizacion - Векторизация
4. The vectorizer converts a word or set of words into a numeric vector that is understandable to a machine learning algorithm that is used to working with numeric tabular data.

Векторизатор преобразует слово или набор слов в числовой вектор, понятный алгоритму машинного обучения, который привык работать с числовыми табличными данными.


Самый простой способ извлечь признаки из текстовых данных -- векторизаторы: CountVectorizer и TfidfVectorizer

The CountVectorizer object does the following thing:

builds for each document (each line that comes to it) a vector of dimension n, where n is the number of words, or n-grams in the entire corpus
fills each i-th element with the number of occurrences of a word in this document.

Объект CountVectorizer делает следующую вещь:

строит для каждого документа (каждой пришедшей ему строки) вектор размерности n, где n -- количество слов или n-грам во всём корпусе
заполняет каждый i-тый элемент количеством вхождений слова в данный документ

# Model - Logistic Regression - RandomForest - XGBoost
