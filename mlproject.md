# Human Activity Recognition

## Summary

The human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time. In the paper "Qualitative Activity Recognition of Weight Lifting Exercises" by E. Velloso, A. Bulling, H. Gellersen, W. Ugulino and H. Fuks it is investigated "how (well)" an activity was performed by the wearer. Six young healthy participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

In this machine learning project we use data from accelerometers on the belt, forearm and dumbbell of the six participants to build a prediction model which is able to discriminate between the five different motions.

## Loading the data


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
importdata <- read.csv("pml-training.csv", header = T)
```


## Preprocessing the data

There are 160 variables in the data. I delete features which can't contribute much to the model (mostly due to missing values). To fit models I change the classes of the predictor variables to "numeric".


```r
data <- importdata[, c(8:10, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160)]
for (i in 1:51) {
    data[, i] <- as.numeric(as.character(data[, i]))
}
```

  
Now, I create a training and a test set. 


```r
inTrain <- createDataPartition(data$classe, p = 0.25, list = F)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
```


Note that my training set is unusually small. However, even with less than 5000 observations the training of the model took more than four hours. And as the test cases weren't too controversial, the resulting model was able to predict all 20 test cases correctly. So, I wasn't inclined to increase the number of observations in the training set or train more models due to the limitations in time and calculating power of my computer.

I would have liked to compare different methods of preprocessing the data by principal component analysis, Box-Cox-transformation, standardizing, or log-transformation. 

## Building the model

Furthermore, I'd like to compare models based on random forests, boosting, k-nearest neighbor, discriminant analysis and support vector machines, and build an ensemble of them which predicts the variable "classe" by majority vote. But again, the time it takes to train a single model hasn't allowed for an exceedingly thorough analysis. The model I trained is based on random forests.

modelrf <- train(classe~., data=training, method="rf")


```r
load("modelrf.rda", verbose = F)
```


## Out of sample error

To see how well the model fits the test set, we can have a look at the ConfusionMatrix..


```r
confusionMatrix(testing$classe, predict(modelrf, testing))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4173    6    4    0    2
##          B   49 2767   21    6    4
##          C    0   40 2507   19    0
##          D    0    0   38 2374    0
##          E    0    5    5   18 2677
## 
## Overall Statistics
##                                         
##                Accuracy : 0.985         
##                  95% CI : (0.983, 0.987)
##     No Information Rate : 0.287         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.981         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.988    0.982    0.974    0.982    0.998
## Specificity             0.999    0.993    0.995    0.997    0.998
## Pos Pred Value          0.997    0.972    0.977    0.984    0.990
## Neg Pred Value          0.995    0.996    0.994    0.997    1.000
## Prevalence              0.287    0.192    0.175    0.164    0.182
## Detection Rate          0.284    0.188    0.170    0.161    0.182
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.994    0.988    0.984    0.990    0.998
```


The Accuracy is 98.5%. That means the out of sample error is 1.5%.

## Application of the model to the test cases

As lined out, I would have liked to play around with other models. Nevertheless, this one is good enough for this project, so let's apply the prediction model to the test cases.


```r
testcases <- read.csv("pml-testing.csv", header = T)
predict(modelrf, testcases)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

