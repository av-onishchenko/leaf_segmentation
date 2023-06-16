Условие: https://drive.google.com/drive/folders/1Nt2poG8rctwWeUfycSSNURe8ryIkMKJc

Ссылка на результаты inference: 

**Семантическая сегментация:** https://github.com/a4-ai/sd23-seg-av-onishchenko/blob/segmentation_task/SemanticSegmentation.ipynb

**Instance сегментация:** https://github.com/a4-ai/sd23-seg-av-onishchenko/blob/segmentation_task/InstanceSegmentation.ipynb

Пайплайн инстанс сегментации:

1. Регрессией предсказываю количество листьев. Беру backbone от Unet обученного на семантическую сегментацию, добавляю еще несколько Res блоков и дообучаю предсказывать количество листьев.
2. Преобразовываю исходную картинку с помощью обученного Unet - на выходе тот же размер, но 10 каналов.
3. Добавляю еще два канала для координат x и y. Таким образом получаю feature extractor для каждого пикселя.
4. Теперь для пикселей сегментированных в задачи семантической сегментации делаю kmeans с параметром найденным регрессией. Результат - предсказанная инстанс маска.
5. Чтобы сравнить две инстанс маски, я считаю всевозможные попарные значения Dice метрики и с помощью венгерского алгоритма нахожу оптимальное соответствие.

Значения метрик:

**DiffFgDICE:** 0.959

**DiffFgMSE:** 0.002

**AbsDiffFGLabels:** 0.524

**DiffFGLabels:** 0.048

**SymmetricBestDice:** 0.531
