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