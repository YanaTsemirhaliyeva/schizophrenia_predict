import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from utils.data_loader import DataLoader
from utils.data_analyzer import DataAnalyzer
from utils.classifier_evaluator import ClassifierEvaluator
from utils.error_handler import ErrorHandler

if __name__ == "__main__":
    try:
        # Загрузка данных
        loader = DataLoader()
        loader.load_csv("data/schizophrenia_dataset.csv")  # Укажите путь к вашему датасету
        loader.data.rename(columns={
            'Hasta_ID': 'Id',
            'Yaş': 'Age',
            'Cinsiyet': 'Gender',
            'Eğitim_Seviyesi': 'Education_Level',
            'Medeni_Durum': 'Martial_Statuts',
            'Meslek': 'Occupation',
            'Gelir_Düzeyi': 'Income_level',
            'Yaşadığı_Yer': 'Live_Area',
            'Tanı': 'Diagnosis',
            'Hastalık_Süresi': 'Disease_Duration',
            'Hastaneye_Yatış_Sayısı': 'Hospitalizations',
            'Ailede_Şizofrenи_Öyküsü': 'Family_History',
            'Madde_Kullanımı': 'Substance_Use',
            'İntihar_Girişimi': 'Suicide_Attempts',
            'Pozitif_Semptom_Skorу': 'Positive_Symptoms_Score',
            'Negatif_Semptом_Sкору': 'Negative_Symptoms_Score',
            'GAF_Sкору': 'Global_Assessment_of_Functioning_Score',
            'Sosyal_Destek': 'Social_Support',
            'Stres_Faktörleri': 'Stress_Factors',
            'İlaç_Uyumu': 'Adherence_to_Medication'
        }, inplace=True)

        # Анализ данных
        loader.count_missing_values()
        analyzer = DataAnalyzer(loader.data)
        analyzer.statistical_summary()
        analyzer.correlation_analysis()

        # Удаление ненужных столбцов
        loader.data.drop(columns=['Id', 'Disease_Duration', 'Hospitalizations'], inplace=True)
        loader.data.drop(
            columns=['Stress_Factors', 'Social_Support', 'Gender', 'Martial_Statuts', 
                     'Occupation', 'Live_Area', 'Age', 'Education_Level', 'Income_level'], 
            inplace=True
        )

        # Построение круговой диаграммы
        counts = loader.data['Suicide_Attempts'].value_counts()
        labels = ['no', 'yes']
        plt.figure(figsize=(6, 6))
        plt.pie(
            counts, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightblue', 'salmon']
        )
        plt.title('Процентное распределение Suicide_Attempts')
        plt.show()

        # Подготовка данных для классификации
        X = loader.data.drop('Diagnosis', axis=1)
        y = loader.data['Diagnosis']

        # Создание объекта для оценки моделей
        evaluator = ClassifierEvaluator(X, y)

        # Оценка отдельных классификаторов
        classifiers = [
            ("Gradient Boosting", GradientBoostingClassifier()),
            ("CatBoost", CatBoostClassifier(verbose=0)),
            ("AdaBoost", AdaBoostClassifier()),
            ("Extra Trees", ExtraTreesClassifier()),
            ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
            ("LightGBM", LGBMClassifier()),
            ("K Neighbors", KNeighborsClassifier()),
            ("Decision Tree", DecisionTreeClassifier()),
            ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
            ("Dummy", DummyClassifier(strategy="most_frequent")),
            ("SVM (Linear Kernel)", SVC(kernel="linear"))
        ]
        evaluator.evaluate(classifiers)

        # Оценка ансамбля Voting
        voting = VotingClassifier(
            estimators=[
                ("Dummy", DummyClassifier(strategy="stratified")),
                ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
                ("CatBoost", CatBoostClassifier(verbose=0))
            ],
            voting="soft"
        )
        evaluator.evaluate([("Voting (Soft)", voting)])

        # Оценка ансамбля Stacking
        stacking = StackingClassifier(
            estimators=[
                ("Dummy", DummyClassifier(strategy="stratified")),
                ("LightGBM", LGBMClassifier()),
                ("CatBoost", CatBoostClassifier(verbose=0))
            ],
            final_estimator=QuadraticDiscriminantAnalysis()
        )
        evaluator.evaluate([("Stacking", stacking)])

        # Вывод лучших классификаторов
        evaluator.get_best_classifier(metric="Accuracy")

    except Exception as e:
        ErrorHandler.log_and_raise(e, "Критическая ошибка в основном потоке")