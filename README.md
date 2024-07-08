# Sentiment-Analysis-Research
Сравнительный анализ методов представления текста и моделей классификации для определения тональности отзывов клиентов онлайн магазина "Ozon". 

Источник отзывов - otzovik.com

Задача определения тональности свелась к определению позитивных/негативных отзывов. Отзывы с оценками 4, 5 получили метку 1 (позитивные), отзывы с оценками 1-3 - метка 0 (негативные). 

В работе рассмотрены статистические методы векторного представления текста 

- Bag of Words
- TF-IDF
- Latent semantic analysis
  
  на основе нейросетей
  
- Word2Vec
- Bert embeddings
  
В качестве моделей бинарной классификации были рассмотреные наивный Байес, логистическая регрессия, случайный лес, бустинги (LightGBM, CatBoost), предобученная модель BERT (ru-bert-tiny2), а также имплементация стекинга с мета-моделью логистической регрессии.

