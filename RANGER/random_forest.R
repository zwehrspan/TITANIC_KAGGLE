library(tidyverse)
library(tidymodels)

set.seed(123)
fullTrainData <- read.csv("../titanic/train.csv")
finalTestData <- read.csv("../titanic/test.csv")
finalTestDataORI <- read.csv("../titanic/test.csv")
pred <-
    c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare")

fullTrainData <- fullTrainData %>%
    mutate(Survived = as.factor(Survived)) %>%
    mutate(Age = case_when(is.na(Age) ~ mean(Age, na.rm = TRUE),
                           TRUE ~ Age)) %>%
    mutate(Embarked = case_when(Embarked == "" ~ "UNKNOWN",
                                TRUE ~ Embarked)) %>%
    mutate(Fare = case_when(is.na(Fare) ~ mean(Fare, na.rm = TRUE),
                            TRUE ~ Fare)) %>%
    select(c("Survived", all_of(pred)))

finalTestData <- finalTestData %>%
    mutate(Age = case_when(is.na(Age) ~ mean(Age, na.rm = TRUE),
                           TRUE ~ Age)) %>%
    mutate(Embarked = case_when(Embarked == "" ~ "UNKNOWN",
                                TRUE ~ Embarked)) %>%
    mutate(Fare = case_when(is.na(Fare) ~ mean(Fare, na.rm =
                                                   TRUE),
                            TRUE ~ Fare)) %>%
    select(c(all_of(pred)))


# Create a recipe for making dummy vars
dataRecipe <-
    recipe(Survived ~ ., data = fullTrainData) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors())

dataSplit <-
    initial_split(fullTrainData, prop = 3 / 4,
                  strata = Survived)

# Create data frames for the two sets:
trainData <- training(dataSplit)
testData  <- testing(dataSplit)

# AttempT to predicted passenger survival
# results: PassengerID, Survived
# Try random forest

rfModelRanger <-
    rand_forest(
        mode = "classification",
        engine = "ranger",
        mtry = 2,
        trees = 1000
    )

dataWF <-
    workflow() %>%
    add_model(rfModelRanger) %>%
    add_recipe(dataRecipe)

rfModelRangeFit <- dataWF %>%
    fit(data = trainData)

#rfModelRangeFit <- rfModelRanger %>%
#    fit(Survived ~ Pclass:Embarked, data = trainData)

# Predict against the training set
rfTrainingPred <-
    predict(rfModelRangeFit, trainData) %>%
    bind_cols(predict(rfModelRangeFit, trainData, type = "prob")) %>%
    bind_cols(trainData %>%
                  select(Survived))

rfTrainingPred %>%                # training set predictions+roc_auc(truth = Survived, .pred_0)
    roc_auc(truth = Survived, .pred_0)
rfTrainingPred %>%                # training set predictions
    accuracy(truth = Survived, .pred_class)
#  Testing Data
rfTestingPred <-
    predict(rfModelRangeFit, testData) %>%
    bind_cols(predict(rfModelRangeFit, testData, type = "prob")) %>%
    bind_cols(testData %>%
                  select(Survived))

rfTestingPred %>%                # testing set predictions
    roc_auc(truth = Survived, .pred_0)

rfTestingPred %>%                # testing set predictions
    accuracy(truth = Survived, .pred_class)


rfTestingPred <-
    predict(rfModelRangeFit, finalTestData)
finalOutput <- cbind.data.frame(finalTestDataORI$PassengerId, rfTestingPred$.pred_class)
colnames(finalOutput) <- c("PassengerId", "Survived")
write_csv(finalOutput, "submission.csv")