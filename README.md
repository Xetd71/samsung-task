# samsung-task
ML task for Samsung internship 2018

## Задача 1

Построить классификатор, разделяющий изображения крокодила и часов

## Отчет

### 1.1

Для предсказания класса по картинке была использована сверточная нейросеть из 12 слоев с использованием библиотеки **Keras** с бзк-эндом **Tensorflow**:<br/>
- 2 слоя свертки
- слой объединения(Pooling Layer)
- отсеивающий слой(от переобучения)(Dropout layer)

- 2 слоя свертки
- слой объединения(Pooling Layer)
- отсеивающий слой(от переобучения)(Dropout layer)

- сглаживающий слой(Flatten layer)
- полносвязный слой
- полносвязный слой с сигмоидой

<br/>**accuracy на обучаемом множестве: 98.95%** и
<br/>**accuracy на тестовом множестве: 92%**
<br/>Пытался еще улучшить результаты, но тщетно. Свалим вину на малый разбер предоставленной выборки, всего 1000 объектов.
<br/>![picture](task1/accuracy-and-loss.png)
<br/>Подробнее см. [task1/train.ipynb](task1/train.ipynb)

### 1.2

Были найдены картинки наиболее похожие и на крокодила, и на часы одновременно, предпологая, что сеть выдает заведомо правильный ответ<br/>
![picture](task1/crock-clock-images.png)
<br/>Подробнее см. [task1/usage.ipynb](task1/usage.ipynb)

### 1.3

Попытался создать картинку максимально похожую и на часы и крокодила, используя **автоенкодер**. Даже получилось обмануть сеть, она выдала вероятность ровно 50% ринадлежности к крокодилу и часам. Но визуально картинка тоже не очень разборчива.<br/>
![picture](task1/generated_image.png)
<br/>Подробнее см. [task1/usage.ipynb](task1/usage.ipynb)

### Использование

Перейдите в папку [task1](task1), запустите команду:
```bash
pip install -r requirements.txt
```
Для тренировки сети можете запустить файл: [train.ipynb](task1/train.ipynb) или [train.py](task1/train.py)
Для использования сети, посика и генерации картинок похожих на крокодила и часы запустите файл: [usage.ipynb](task1/usage.ipynb) или [usage.py](task1/usage.py)

## Задача 2

Построить классификатор, который может определить кому принадлежит данный
отрезок текста: Гоголю или Гегелю.

## Отчет

Для классификации было решено использовать **логистическую регрессию** в связке с **tf-idf** кодированием для одного слова и пары слов, поскольку для классификации текста на 2 кластера этого вполне достаточно и точно не нужно усложнять модель нейросетями.

### Подготовка данных

Были выбраны произольные произведения Гоголя и Гелеля, которые в дальнецшем делились на небольшие выборки по несколько предложений. Далее выборки:<br/>
- приводились к нижнему регистру
- убиралась пунктуация
- фильтровались стоп слова и слова не из алфавита
- все слова в выборке преобразовывались в коренные

<br/>Позже данные сохранялись в файл [data.csv](task2/data/data.csv)
<br/>Подробнее см. [task2/prepare_data.ipynb](task2/prepare_data.ipynb)

### Модель

<br/>Данные из [data.csv](task2/data/data.csv) преобразовывались в **tf-idf** матрицу.
<br/>Далее с помощью **GridSearchCV** находились лушкие параметры для логистической регрессии с L2 регулязацией.
<br/>После этого модель училась и тестировалась.

<br/>**ROC-AUC на обучаемом множестве: 99.61%** и
<br/>**ROC-AUC на тестовом множестве: 99.75%**

<br/>ROC-AUC выглядит не очень интересно<br/>
![picture](task2/roc-auc-curve.png)
<br/>На отрывках других произведений модель также имела отличные предсказания
<br/>Подробнее см. [task2/train.ipynb](task2/train.ipynb)

### Использование

Перейдите в папку [task2](task2), запустите команду:
```bash
pip install -r requirements.txt
```
Для подготовки ланных можете запустить файл: [prepare_data.ipynb](task2/prepare_data.ipynb) или [prepare_data.py](task2/prepare_data.py)
Для тренировки сети можете запустить файл: [train.ipynb](task2/train.ipynb) или [train.py](task2/train.py)


