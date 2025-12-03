# Description 


# Taks
1. Ejecute todas las celdas y obtenga los resultados

2. Proporcione los resultados de la tabla classification_report debajo de esta tarea para el modelo LogisticRegression.

3. Aplique 2 algoritmos alternativos al utilizado para resolver el problema de clasificación (por ejemplo, XGBClassifier y algún otro más) y obtenga los resultados en la tabla classification_report.

4. Para XGBClassifier necesitará definir los parámetros:
learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27

5. En la sección de vectorización TF-IDF, de manera análoga a los unigramas y pentagramas, calcule el classification_report para bigramas y trigramas. Publique los resultados en el informe e indique si cambió la precisión (f1-score) al usarlos en comparación con los unigramas y pentagramas.


Pasos en el proceso:
1. Cargar los CSV en pandas y combinar/etiquetar.
2. Preprocesar texto (limpieza básica).
3. Vectorizar texto (por ejemplo, TF-IDF).
4. Entrenar un clasificador LogisticRegression.
5. Predecir sobre conjunto de prueba.
6. Mostrar classification_report (precision, recall, f1-score y support).


Concepts
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

