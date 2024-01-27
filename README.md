# Handwritten Persian Digits Recognition with K-Nearest Neighbors

This repository contains a Python script for recognizing handwritten persian digits using the K-Nearest Neighbors (KNN) algorithm. The dataset used for training and testing is the Hoda dataset, which consists of handwritten digits.

## Contents

- `knn_digits_recognition.py`: Python script that performs the following tasks:
  - Loads the Hoda dataset using scipy.
  - Prepares the dataset for training and testing by resizing and reshaping the images.
  - Utilizes the K-Nearest Neighbors (KNN) classifier from Scikit-Learn to train and test the model.
  - Prints predictions for individual samples and calculates the accuracy of the model.

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install numpy scipy opencv-python scikit-learn matplotlib
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/hkhoshraftar2/ML-persian-digits-classification.git
cd ML-persian-digits-classification
```

2. Run the Python script:

```bash
python persian-digit-classification.py
```

3. Explore the output, including individual predictions and the overall accuracy of the model.

## Example

```python
# Example usage for predicting a single sample
sample = 24
X = [X_test[sample]]
predicted_class = neigh.predict(X)
print("Sample {} is a {}, and your prediction is: {}.".format(sample, y_test[sample], predicted_class[0]))

# Example usage for printing probably neighbor values
print(neigh.predict_proba(X))

# Example usage for calculating accuracy
acc = neigh.score(X_test, y_test)
print("Accuracy is %.2f %%" % (acc * 100))
```
Techniques Used
Dataset Loading: The script loads the Hoda dataset using Scipy's io module.

Data Preprocessing: It prepares the dataset for training and testing by extracting the necessary features and labels.

Image Resizing: The images in the dataset are resized using OpenCV's cv2.resize to a specified size (e.g., 5x5 pixels).

Data Reshaping: Reshaping is performed to convert the 2D images into 1D arrays for training and testing.

K-Nearest Neighbors Classifier: Scikit-Learn's KNeighborsClassifier is employed to build and train the machine learning model.

Model Testing: The trained model is tested on a subset of the dataset, and individual predictions are printed for a sample.

Probability Prediction: The script prints the probability values for each class using predict_proba.

Accuracy Calculation: The accuracy of the model is calculated and printed, providing an indication of the model's performance.

# تشخیص اعداد دستنویس فارسی با استفاده از روش K-Nearest Neighbors

## مقدمه

این مخزن شامل یک اسکریپت پایتون برای تشخیص اعداد دستنویس فارسی با استفاده از الگوریتم K-Nearest Neighbors (KNN) است. مجموعه داده مورد استفاده برای آموزش و آزمایش، مجموعه داده Hoda است که شامل اعداد دستنویس می‌باشد.

## محتوا

- `knn_digits_recognition.py`: اسکریپت پایتون که وظایف زیر را انجام می‌دهد:
  - بارگذاری مجموعه داده Hoda با استفاده از scipy.
  - آماده‌سازی مجموعه داده برای آموزش و آزمایش با تغییر اندازه و تغییر شکل تصاویر.
  - استفاده از طبقه‌بند K-Nearest Neighbors (KNN) از کتابخانه Scikit-Learn برای آموزش و آزمایش مدل.
  - چاپ پیش‌بینی‌ها برای نمونه‌های مجزا و محاسبه دقت مدل.

## نیازمندی‌ها

اطمینان حاصل کنید که نیازمندی‌های زیر نصب شده باشند:

```bash
pip install numpy scipy opencv-python scikit-learn matplotlib
```

## راهنمای استفاده

1. این مخزن را کلون کنید:

```bash
git clone https://github.com/hkhoshraftar2/ML-persian-digits-classification.git
cd ML-persian-digits-classification
```

2. اسکریپت پایتون را اجرا کنید:

```bash
python persian-digit-classification.py
```

3. خروجی را بررسی کنید، شامل پیش‌بینی‌های نمونه‌ها و دقت کل مدل.

## نمونه

```python
# نمونه استفاده برای پیش‌بینی یک نمونه
sample = 24
X = [X_test[sample]]
predicted_class = neigh.predict(X)
print("نمونه {} یک {}, و پیش‌بینی شما: {} است.".format(sample, y_test[sample], predicted_class[0]))

# نمونه استفاده برای چاپ احتمالات همسایگی
print(neigh.predict_proba(X))

# نمونه استفاده برای محاسبه دقت
acc = neigh.score(X_test, y_test)
print("دقت %.2f %% است" % (acc * 100))
```

## تکنیک‌های استفاده شده

1. **بارگذاری مجموعه داده**: این اسکریپت مجموعه داده Hoda را با استفاده از ماژول `io` در scipy بارگذاری می‌کند.

2. **پیش‌پردازش داده**: این مجموعه داده را برای آموزش و آزمایش با استخراج ویژگی‌ها و برچسب‌های لازم آماده می‌کند.

3. **تغییر اندازه تصویر**: تصاویر در مجموعه داده با استفاده از `cv2.resize` در OpenCV به اندازه مشخص (مثلاً 5x5 پیکسل) تغییر اندازه می‌شوند.

4. **تغییر شکل داده**: تغییر شکل انجام می‌شود تا تصاویر 2D به آرایه‌های 1D برای آموزش و آزمایش تبدیل شوند.

5. **طبقه‌بند K-Nearest Neighbors**: از `KNeighborsClassifier` در Scikit-Learn برای ساخت و آموزش مدل ماشینی استفاده می‌شود.

6. **آزمایش مدل**: مدل آموزش داده شده روی یک زیرمجموعه از مجموعه داده آزمایش شده و پیش‌بینی‌های جداگانه برای یک نمونه چاپ می‌شود.

7. **پیش‌بینی احتمالاتی**: این اسکریپت احتمالات برای هر کلاس را با استفاده از `predict_proba` چاپ می‌کند.

8. **محاسبه دقت**: دقت مدل محاسبه و چاپ می‌شود که نشان دهنده عملکرد مدل است.

احساس راحتی کنید که اسکریپت را بررسی و تغییر دهید تا به نیازهای خاص خودتان بپردازید. مجموعه داده از

