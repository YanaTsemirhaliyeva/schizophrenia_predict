#!/usr/bin/env python
# coding: utf-8

import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay)
from utils.error_handler import ErrorHandler
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Логирование в файл
        logging.StreamHandler()         # Логирование в консоль
    ]
)

class ClassifierEvaluator:
    def __init__(self, X: pd.DataFrame, y: pd.Series, output_dir='public'):
        """
        Инициализация оценщика классификаторов.

        :param X: Фичи (признаки) датасета.
        :param y: Целевая переменная.
        """
        self.X = X
        self.y = y
        self.results = []
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Создана директория для сохранения графиков: {self.output_dir}")

    def evaluate(self, classifiers: list) -> None:
        """
        Оценивает переданные классификаторы на тренировочных данных.

        :param classifiers: Список кортежей вида [("Название", объект_классификатора), ...].
        """
        try:
            if not classifiers:
                ErrorHandler.log_warning("Список классификаторов пуст. Нечего оценивать.")
                return
            
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            logging.info("Начало оценки переданных классификаторов...")
            for name, classifier in classifiers:
                logging.info(f"Оценка классификатора: {name}")
                classifier.fit(X_train, y_train)  # Обучение на тренировочных данных
                y_pred = classifier.predict(X_test)  # Предсказания на тестовых данных

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")

                # Вывод результатов
                print(f"Результаты для классификатора {name}:")
                print(classification_report(y_test, y_pred))  # Подробный отчёт
                print(f"Accuracy: {accuracy * 100:.2f}%")
                
                # Отображение матрицы ошибок
                ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
                plt.title(f"Confusion Matrix for {name}")  # Добавить подпись
                plt.savefig(f'{self.output_dir}/{name}.png', format='png', dpi=300, bbox_inches='tight')
                plt.show()

                logging.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                self.results.append({
                    "Classifier": name,
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall
                })

            logging.info("Оценка классификаторов завершена.")
        except Exception as e:
            ErrorHandler.log_and_raise(e, "Ошибка при оценке классификаторов")

    def get_best_classifier(self, metric: str = "Accuracy") -> None:
        """
        Выводит лучший классификатор по заданной метрике.

        :param metric: Метрика для сравнения (Accuracy, F1 Score, Precision, Recall).
        """
        try:
            if not self.results:
                ErrorHandler.log_warning("Нет результатов оценки. Выполните метод evaluate() перед вызовом get_best_classifier().")
                return

            # Преобразуем результаты в DataFrame
            df_results = pd.DataFrame(self.results)

            # Находим максимальное значение метрики
            max_metric = df_results[metric].max()

            # Фильтруем классификаторы, у которых метрика равна максимальному значению
            best_classifiers = df_results[df_results[metric] == max_metric]

            logging.info(f"Лучшие классификаторы по метрике {metric}:")
            logging.info(f"\n{best_classifiers}")

        except Exception as e:
            ErrorHandler.log_and_raise(e, "Ошибка при выборе лучшего классификатора")