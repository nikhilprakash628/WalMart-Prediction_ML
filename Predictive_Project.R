# Load Libraries
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(ggplot2)

# Load the Walmart dataset
walmart_data <- read.csv(file.choose())
View(walmart_data)
str(walmart_data)

# Preprocessing
# Convert `Date` to Date type and extract year, month, and week
walmart_data$Date <- as.Date(walmart_data$Date, format = "%d-%m-%Y")
walmart_data$Year <- as.numeric(format(walmart_data$Date, "%Y"))
walmart_data$Month <- as.numeric(format(walmart_data$Date, "%m"))
walmart_data$Week <- as.numeric(format(walmart_data$Date, "%U"))

# Create a binary target variable: High or Low Weekly Sales based on median
median_sales <- median(walmart_data$Weekly_Sales)
walmart_data$sales_category <- ifelse(walmart_data$Weekly_Sales > median_sales, "High", "Low")
walmart_data$sales_category <- as.factor(walmart_data$sales_category)

# Remove the original `Weekly_Sales` column for classification tasks
walmart_data <- walmart_data[, !names(walmart_data) %in% c("Weekly_Sales", "Date")]

# Normalize numerical features
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
numerical_columns <- c("Temperature", "Fuel_Price", "CPI", "Unemployment")
walmart_data[, numerical_columns] <- lapply(walmart_data[, numerical_columns], normalize)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(walmart_data$sales_category, p = 0.8, list = FALSE)
train_data <- walmart_data[trainIndex, ]
test_data <- walmart_data[-trainIndex, ]

# Store accuracies for comparison
accuracy_results <- data.frame(Model = character(), Accuracy = numeric())

# 1. Logistic Regression
log_model <- glm(sales_category ~ ., data = train_data, family = "binomial")
log_prob <- predict(log_model, test_data, type = "response")
log_class <- as.factor(ifelse(log_prob > 0.5, "High", "Low"))
cm_log <- confusionMatrix(log_class, test_data$sales_category)
accuracy_log <- cm_log$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Logistic Regression", Accuracy = accuracy_log))

# Plot: Logistic Regression Predicted Probabilities
ggplot(data.frame(Actual = test_data$sales_category, Predicted_Prob = log_prob), aes(x = Actual, y = Predicted_Prob, fill = Actual)) +
  geom_boxplot() +
  labs(title = "Logistic Regression: Predicted Probabilities", x = "Actual Outcome", y = "Predicted Probability") +
  theme_minimal()

# 2. Naive Bayes
nb_model <- naiveBayes(sales_category ~ ., data = train_data)
nb_predictions <- predict(nb_model, test_data)
cm_nb <- confusionMatrix(nb_predictions, test_data$sales_category)
accuracy_nb <- cm_nb$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Naive Bayes", Accuracy = accuracy_nb))

# Plot: Naive Bayes Predicted Probabilities
nb_prob <- predict(nb_model, test_data, type = "raw")
nb_prob_df <- data.frame(Actual = test_data$sales_category, Prob_High = nb_prob[, "High"], Prob_Low = nb_prob[, "Low"])
ggplot(nb_prob_df, aes(x = Actual)) +
  geom_boxplot(aes(y = Prob_High, fill = "High"), alpha = 0.5) +
  geom_boxplot(aes(y = Prob_Low, fill = "Low"), alpha = 0.5) +
  labs(title = "Naive Bayes: Predicted Probabilities", x = "Actual Outcome", y = "Predicted Probability", fill = "Class") +
  theme_minimal()

# 3. Support Vector Machine (SVM)
svm_model <- svm(sales_category ~ ., data = train_data, kernel = "linear")
svm_predictions <- predict(svm_model, test_data)
cm_svm <- confusionMatrix(svm_predictions, test_data$sales_category)
accuracy_svm <- cm_svm$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "SVM", Accuracy = accuracy_svm))

# Plot: SVM Predicted Classes
ggplot(data.frame(Actual = test_data$sales_category, Predicted = svm_predictions), aes(x = Actual, y = Predicted, fill = Actual)) +
  geom_jitter(alpha = 0.6, size = 3) +
  labs(title = "SVM: Actual vs Predicted Classes", x = "Actual Outcome", y = "Predicted Outcome") +
  theme_minimal()

# 4. Decision Tree
tree_model <- rpart(sales_category ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, test_data, type = "class")
cm_tree <- confusionMatrix(tree_predictions, test_data$sales_category)
accuracy_tree <- cm_tree$overall["Accuracy"]
accuracy_results <- rbind(accuracy_results, data.frame(Model = "Decision Tree", Accuracy = accuracy_tree))

# Plot: Decision Tree
rpart.plot(tree_model, main = "Decision Tree")

# Final Accuracy Plot
ggplot(accuracy_results, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(Accuracy, 2)), vjust = -0.5, size = 4) +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal()

