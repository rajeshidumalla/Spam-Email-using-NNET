## Libraries Import

```{r}
# Importing libraries and set up the work environment
library(knitr)
library(readxl)
library(tidyverse)
library(janitor)
library(rpart)
library(nnet)
library(scales)
```


```{r}
knitr::opts_chunk$set(cache = TRUE,
warning = FALSE,
echo = TRUE,
message = FALSE,
dpi = 180,
fig.width = 6,
fig.height = 4)
theme_set(theme_classic())
```

## Data Import
Loading the data files, assigning the header names and counting to check fro class imabalance.

```{r}
spam_train <- read_csv("spam_stats315B_train.csv",
col_names = FALSE)
spam_test <- read_csv("spam_stats315B_test.csv",
col_names = FALSE)
header_spam <- c("make", "address", "all", "3d", "our", "over", "remove",
"internet","order", "mail", "receive", "will",
"people", "report", "addresses","free", "business",
"email", "you", "credit", "your", "font","000","money",
"hp", "hpl", "george", "650", "lab", "labs",
"telnet", "857", "data", "415", "85", "technology", "1999",
"parts","pm", "direct", "cs", "meeting", "original", "project",
"re","edu", "table", "conference", ";", "(", "[", "!", "$", "#",
"CAPAVE", "CAPMAX", "CAPTOT","type")
colnames(spam_train) <- header_spam
colnames(spam_test) <- header_spam
# Checking for class imbalance
spam_train %>%
count(type)
```
## Accuracy Functions

```{r}
# Important Functions
pct_accuracy <- function(y_hat, y_test, threshold){
y_hat[y_hat > threshold] <- 1
y_hat[y_hat <= threshold] <- 0
correct <- y_hat == y_test
pct_accurate <- sum(correct)/length(correct)
return(pct_accurate)
}

# Missclasification RAtes - Missed in previous HWs
get_misclassification_rates <- function(model, threshold){
  y_hat <- predict(model, spam_test_proc)
y_hat[y_hat > threshold] <- 1
y_hat[y_hat <= threshold] <- 0
correct <- y_hat == y_test
correct_spam <- correct[y_test == 1]
correct_nonspam <- correct[y_test == 0]
misclassification_rate <- 1 - sum(correct)/length(correct)
spam_misclassification_rate <- 1 - sum(correct_spam)/length(correct_spam)
nonspam_misclassification_rate <- 1 - sum(correct_nonspam)/length(correct_nonspam)

return(c(misclassification_rate,
spam_misclassification_rate,
nonspam_misclassification_rate))
}
```

## Pre-Processing

Scaling the data using tidymodels and splitting into training and testing set. Included an example of data
pre-processed.

```{r}
set.seed(415)
# Scaling Data using Tidymodels
spam_train_rec <- recipe(type ~ ., data = spam_train) %>%
step_scale(all_predictors()) %>% # Normaliing the data
step_zv(all_predictors()) %>% # Eliminating Varibales with zero variance
prep(retain = TRUE)
```

## Data Split

```{r}
# Response label
y_test <- spam_test %>%
select(type)
y_train <- spam_train %>%
select(type)
# Splitting data
spam_train_proc <- bake(spam_train_rec, new_data = spam_train) %>%
select(-type)
spam_test_proc <- bake(spam_train_rec, new_data = spam_test) %>%
select(-type)
head(spam_train_proc, 10)
```

(a) Fit on the training set one hidden layer neural networks with 1, 2,…, 10 hidden units and
different sets of starting values for the weights (obtain in this way one model for each number
of units). Which structural model performs best at classifying on the test set?

## First Model - Hidden Neural Layers

```{r}
set.seed(415)
# Parameters
num_neurons <- seq(1:10)
num_reps <- 10
wt_rang = 0.3
threshold <- 0.5
accuracies <- c()


for(size in num_neurons){
  sum_accuracy <- 0
  
  
  for(i in c(1:num_reps)){
    
    set.seed(415)
    model <- nnet(
      spam_test_proc, y_train, size=num_neurons[size],
      linout = FALSE, softmax = FALSE,
      censored = 100, rang = wt_rang, decay = 0,
      maxit = 100, trace = FALSE, Hess = FALSE
    )
    
    y_hat <- predict(model, spam_test_proc)
    sum_accuracy <- sum_accuracy + pct_accuracy(y_hat, y_test, threshold)
  }
  accuracies <- c(accuracies, sum_accuracy/num_reps)
}


#get the num_neurons corresponding with model that produced highest average accuracy
best_performing <- which.max(accuracies)
best_num_neurons <- num_neurons[best_performing]
```

The best number of hidden layer neurons is 4 which provides an accuracy of 94% in the testing set.

Choose the optimal regularization (weight decay for parameters 0,0.1,…,1) for the structural
model found above by averaging your estimators of the misclassification error on the test
set. The average should be over 10 runs with different starting values. Describe your final
best model obtained from the tuning process: number of hidden units and the corresponding
value of the regularization parameter. What is an estimation of the misclassification error of
your model?

## Second Model - Weight Decays

```{r}
set.seed(415)
# Parameters
decays <- seq(0, 1, .1) # Adding weight decay (0.1, 1)
num_neurons <- seq(1:10) # Same layer as before
num_reps <- 10
wt_rang = 0.5
threshold <- 0.5
accuracies <- c()
for(decay in decays){
sum_accuracy <- 0
for(i in c(1:num_reps)){
set.seed(415)
model <- nnet(
spam_train_proc, y_train, size=best_num_neurons,
linout = FALSE, softmax = FALSE,
censored = FALSE, skip = FALSE, rang = wt_rang, decay = decay ,
maxit = 100, Hess = FALSE, trace = FALSE
)
y_hat <- predict(model, spam_test_proc)
sum_accuracy <- sum_accuracy + pct_accuracy(y_hat, y_test, threshold)
}
accuracies <- c(accuracies, sum_accuracy/num_reps)
}
# Highest average accuracy by neuron and decay.
best_performing <- which.max(accuracies)
best_decay <- decays[best_performing]
```

The best performing model is achieved using 0.1 as the best weight decay.

### Missclasification rate using both parameters Number of Neurons and Weight Decay.

```{r}
#get misclassification rates for our best chosen parameters
model <- nnet(
spam_train_proc, y_train, size= best_num_neurons,
linout = FALSE, entropy = FALSE, softmax = FALSE,
censored = FALSE, skip = FALSE, rang = wt_rang, decay = best_decay,
maxit = 100, Hess = FALSE, trace = FALSE
)
misclassification_rates <- get_misclassification_rates(model, threshold)#get misclassification rates for our best chosen parameters
```

Best Weight Decay: 0.1
Best Number of Layer Neurons: 4
Missclasification rate of the model: 4%
Spam misscalsification rate: 6%
Ham Missclasification Rate: 3%
(c) As in the previous homework

misclassified good emails to be less than 1%.

```{r}
set.seed(415)
#for a given model
find_threshold <- function(model){
thresholds <- seq(0,1,0.01)
y_hat <- predict(model, spam_test_proc)
for(thresh in thresholds){
y_hat_nonspam <- y_hat[y_test == 0]
y_hat_nonspam[y_hat_nonspam > thresh] <- 1
y_hat_nonspam[y_hat_nonspam <= thresh] <- 0
nonspam_misclassification_rate <- sum(y_hat_nonspam)/length(y_hat_nonspam)
if(nonspam_misclassification_rate <= 0.01) {break}
}
return(thresh)
}
```

```{r}
# parameters
num_neurons <- seq(1:10)
weight_decays <- seq(0,1,.1)
wt_rang = 0.5
#fitting the modee to specified parameters
models <- list()
for(i in c(1:length(num_neurons))){
for(j in c(1:length(decays))){
model <- nnet(
spam_train_proc, y_train, size=num_neurons[i],
linout = FALSE, entropy = FALSE, softmax = FALSE,
censored = FALSE, skip = FALSE, rang = wt_rang, decay = decays[j],
maxit = 100, Hess = FALSE, trace = FALSE
)
models[[paste(i,j,sep="_")]] <- model
}
}
# < 1% nonspam misclassification rate for each model
model_thresholds <- lapply(models, function(x) {find_threshold(x)})
#get the overall, spam, and non-spam misclassification rates at each threshold
misclassification_rates <- mapply(get_misclassification_rates, models, model_thresholds)
#find the model with the lowest overall misclassification rate
#(using forced < 1% nonspam threshold)
best_model_idx <- which.min(misclassification_rates[1,])
best_model <- models[[best_model_idx]]
best_misclassification_rates <- misclassification_rates[,best_model_idx]
```

Reporting on the best neural net model:

Best model accuracy for Ham emails: (< 1% missclasification)

Hidden layer neurons: 8 Weight Decay: 0.3

Overall Missclasification Rate: 5%

Spam Email Missclasification Rate: 11%

Ham Email Missclasification Rate: 1%
