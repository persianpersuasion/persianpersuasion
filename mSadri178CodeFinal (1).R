#ECON 178 Final Project Code
#Setting my working directory to where variables are stored

#Part 1: Feature Variable Inspection & Preliminary Findings

setwd("/Users/mehrisadri/Downloads")
#Now, we will load the full dataset
data_tr <- read.table("data_tr.txt", header = TRUE, sep = "\t", dec = ".")[,-1]
#Verifying that we loaded the correct dataset
head(data_tr)
#Now, I will remove some columns (available predictors) I don't want involved
data_tr_fin <- data_tr[, !names(data_tr) %in% c("hequity", "nohs", "smcol")]
names(data_tr_fin)
#Now, I will get a general summary of the dataset before taking a closer inspection
# of certain variables to transform plain predictors so that our model
# runs smoother
summary(data_tr_fin)
#Observing our Y value
hist(data_tr_fin$tw, breaks=100, main="histogram of Total Wealth", xlab="Total Wealth")
#Scatterplots
library(ggplot2)
plot(data_tr$tw, data_tr$age, 
     xlab = "Total Wealth (USD)", 
     ylab = "Age (in Years)", 
     main = "Age vs. Total Wealth")
#We see above that there is a greater number of individuals ages 40 and up that have ammased greater wealth
#I have an intuitive understanding above how some variables change in relation to our Y
# value, but I am less sure about others. Thus, I will start with visualizing these 
# nebulous datapoints 
scatter_plot_hmort <- ggplot(data_tr, aes(x = hmort, y = tw)) + 
  geom_point(alpha = 0.7) +
  theme_minimal() +
  labs(title = "Total Wealth vs. Home Mortgage",
       x = "Home Mortgage", 
       y = "Total Wealth (USD)")
print(scatter_plot_hmort)

#Adding a line

ggplot(data_tr, aes(x = hmort, y = tw)) + 
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") + 
  theme_minimal() +
  labs(title = "Total Wealth vs. Home Mortgage",
       x = "Home Mortgage (hmort)", 
       y = "Total Wealth (tw)")

#The result of this scatterplot is not what I expected; as there is a large
# number of individuals with a hmort of 0 that still have considerably higher
# wealth than those with very large home mortgage payments

#Now, let's check ira and non-401k financial assetts
scatter_plot_nifa <- ggplot(data_tr, aes(x = nifa, y = tw)) + 
  geom_point(alpha = 0.7) +
  theme_minimal() +
  labs(title = "Total Wealth vs. Net Financial Assets",
       x = "Net Financial Assets (nifa)", 
       y = "Total Wealth (tw)")
print(scatter_plot_nifa)

scatter_plot_ira <- ggplot(data_tr, aes(x = ira, y = tw)) + 
  geom_point(alpha = 0.7) +
  theme_minimal() +
  labs(title = "Total Wealth vs. Individual Retirement Account",
       x = "Individual Retirement Account (ira)", 
       y = "Total Wealth (tw)")
print(scatter_plot_ira)

#With so many observations (rows), it is difficult to 
#visualize a clear relationship between tw and ira/nifa


#Now, to get an even better understanding on these features, I will fit a linear
# regression model on these three features to see the effect on tw from changes
# to hmort, nifa, and ira
lm_hmort <- lm(tw ~ hmort, data = data_tr)
lm_nifa <- lm(tw ~ nifa, data = data_tr)
lm_ira <- lm(tw ~ ira, data = data_tr)

summary(lm_hmort)
#The wide range of values shows that there is room for prediction error
#through using just this model. For every 1 dollar increase in total wealth,
#home mortgage value raises by ~ .75. However, it is clear that we are still
#missing important predictors
summary(lm_nifa)
#The coefficient is positive and significant. For every 1 dollar increase in
# net financial assets, total wealth increases by 1 dollar and 48 cents. The 
# R-squared value is much larger than for hmort, meaning that nifa is a more important 
# predictor than hmort. This is good to know.
summary(lm_ira)
#The coefficient tells us that for every 1 dollar increases in the ira variable,
# total wealth increases by 5 dollars and 60 cents, which is our highest slope
# of the three. The R-squared value is around 25 percent, which is a better
# indicator than hmort but not nifa. 

#All three of these calculations are statistically significant
#Now, let's look at the combined effect on total wealth from these three variables
#We will assume they are not multicollinear
model <- lm(tw ~ nifa + hmort + ira, data = data_tr)
summary(model)
#With these three variables together, we see that ira is still the steepest
# slope, with an R-Squared of 61 showing that these three factors account
# for a large portion of total wealth. They will not be dropped from the
# final predictor list unless they are multicollinear

#To further identify which predictors may be most significant, I will now run a forward/backwards
#stepwise regression. We will start from 1) analysis from all predictors and 2)analysis from only one predictor
install.packages("MASS")
library(MASS)
null <- lm(tw ~ 1, data = data_tr)
full <- lm(tw ~ ., data = data_tr)
forward_reg <- stepAIC(null, scope = list(lower = null, upper = full), trace = FALSE, direction = 'forward')

#Backwards regression
backward_reg <- stepAIC(full, scope = list(lower = null, upper = full), trace = FALSE, direction = 'backward')
forward_coef <- forward_reg$coefficients
backward_coef <- backward_reg$coefficients
full_coef <- full$coefficients

#Checking the values
summary(forward_reg)
summary(backward_reg)
#Variables like male and twoearn are the least statistically significant,
# and both regressions show similar, statistically significant results.
#Thus, I am inclined to remove male and twoearn from my final predictors
#list

#Now, I am testing for collinear or multicollinear values by using dummy variables/regression
#We need dummy variables to run regressions/for OLS, so it's important to 
#convert the categorical variables now.
data(Wage, package="ISLR")
# We transformed the matrix into a set of 0/1 dummy variables. Now every component is numeric!
new_X <- model.matrix(~. -1, data = Wage[, c(1:9)])
summary(Wage)
#Running regression model to identify multicollinear values
OLS <- lm(tw ~ ., data = data_tr)
summary(OLS)
#After viewing the output, it is clear there is no multicollinearity. If there was,
#there would be NA values.
#None of the variables I decided to keep are multicollinear, so I will not be
# dropping any predictors before beginning OLS/Ridge Regression

#Based on the analysis conducted above, I will be keeping the variables
#I originally chose except for male and twoearn after seeing my
# results from the forward/backward regression.
data_tr_fin <- data_tr[, !names(data_tr) %in% c("twoearn", "man")]
names(data_tr_fin)

#Part 2: Prediction Performance and Testing
#Ridge Regression
#First, I will start with importing the needed library
library(glmnet)

#Now I will define the parameters for the model
y <- data_tr$tw
X <- as.matrix(data_tr[,-1])

#Specifying the range (parameters) of Lambda -- [-20,20] for high variance
lambdas.rr <- exp(seq(-20, 20, length = 100))

set.seed(10)

#Set alpha equal to 0

ridge_cv <- cv.glmnet(x = X, y = y, lambda = lambdas.rr,alpha = 0)

#Seeing what the value of lambda is and getting the final regression values
ridge_cv$lambda.min
#Viewing the lambda value graphically
plot(ridge_cv)
#Based on the visual, it is clear that MSE remains lowest at ~6 before increasing, thus
#I feel comfortable using such a value to prevent over/underfitting

ridge <- glmnet(x = X, y = y, lambda = ridge_cv$lambda.min,alpha = 0)
#Viewing the weights assigned to each predictor
ridge$beta

#Now, it's time to split, train, and test the data

n <- length(y)
set.seed(10)
percent <- 0.8
train <- sample(1:n < percent * n)
test <- !train

#Preparing data format
X.train <- as.matrix(X[train, ])
y.train <- y[train]
X.test <- as.matrix(X[test, ])
y.test <- y[test]

#We already performed cross-validation through ridge_cv
ridge <- glmnet(x = X.train, y = y.train, lambda = ridge_cv$lambda.min, alpha = 0)

#Make predictions
pr.ridge <- predict(ridge, newx = X.test)

#Calculate MSPE
mspe_ridge <- mean((y.test - pr.ridge)^2)
print(paste("Ridge MSPE:", round(mspe_ridge, 4)))

#Now, let's see how a stepwise regression in tandem with our ridge regression affects performance
#I will be using the code provided in lecture, without considering the Lasso model
#The goal is to use the stepwise regression for feature selection (as done during EDA),
#then use the ridge regression to prevent over/underfitting

n <- length(y)
k <- 5
ii <- sample(rep(1:k, length= n))
pr.stepwise_backward <- pr.stepwise_forward <- pr.ridge <- rep(NA, length(y))

for (j in 1:k) {
  hold <- (ii == j)
  train <- (ii != j)
  
  # Stepwise Regression
  full <- lm(tw ~ ., data = data_tr_fin[train, ])
  null <- lm(tw ~ 1, data = data_tr_fin[train, ])
  a <- stepAIC(null, scope = list(lower = null, upper = full), direction = 'forward')
  b <- stepAIC(full, scope = list(lower = null, upper = full), direction = 'backward')
  
  pr.stepwise_backward[hold] <- predict(b, newdata = data_tr_fin[hold, ])
  pr.stepwise_forward[hold] <- predict(a, newdata = data_tr_fin[hold, ])
  
  # Ridge Regression
  xx.tr <- as.matrix(data_tr_fin[train, -1])
  y.tr <- y[train]
  xx.te <- as.matrix(data_tr_fin[hold, -1])
  
  ridge.cv <- cv.glmnet(x = xx.tr, y = y.tr, nfolds = k, alpha = 0)
  pr.ridge[hold] <- predict(ridge.cv$glmnet.fit, newx = xx.te, s = ridge.cv$lambda.min)
}

mspe_step_backward <- mean((pr.stepwise_backward - y)^2, na.rm = TRUE)
mspe_step_forward <- mean((pr.stepwise_forward - y)^2, na.rm = TRUE)
mspe_ridge <- mean((pr.ridge - y)^2, na.rm = TRUE)

#Comparing the MSPE values:
print(mspe_step_backward) #Solely on backwards stepwise
print(mspe_step_forward) #Solely on forward stepwise
print(mspe_ridge) #Solely on ridge regression

#Now, let's use our stepwise + ridge regression model to output prediction
selected_features <- names(coef(b))[-1] 
X.selected <- as.matrix(data_tr_fin[, selected_features])

n <- length(y)
set.seed(10)
percent <- 0.8
train <- sample(1:n, percent * n)
test <- setdiff(1:n, train)

X.train <- X.selected[train, ]
y.train <- y[train]
X.test <- X.selected[test, ]
y.test <- y[test]

#Perform regression and make predictions on test set
ridge_selected <- glmnet(x = X.train, y = y.train, lambda = ridge_cv$lambda.min, alpha = 0)
pr.ridge_selected <- predict(ridge_selected, newx = X.test)

mspe_ridge_selected <- mean((y.test - pr.ridge_selected)^2)
print(paste("Ridge MSPE on Stepwise Selected Features:", round(mspe_ridge_selected, 4)))

#Visualization for paper
plot(y.test, pr.ridge_selected, main = "Predictions vs Actual (Ridge on Stepwise Features)",
     xlab = "Actual", ylab = "Predicted", col = "red", pch = 16)
abline(a = 0, b = 1, col = "blue")


#Since a lower MSPE indicates better model performance, we can conclude that the forward
# and backward stepwise regression work equally well, while the ridge regression
# works slightly worse > MSPE

#The following output shows that when "male" and "marr" are not included, the model
# becomes slightly better

#Age, Ira, hmort, and nifa seem to matter the most

#Now, I will run a (simplier) regression -- OLS -- and compare MSPE

regression3 <- lm(tw ~ ., data=data_tr)
summary(regression3)

#Taking the training dataset made earlier
ols_pred <- predict(reg4, newdata = data_tr)

#Computing MSE to compare with MSPE of Ridge, and MSPE of Stepwise + Ridge Regression
MSE <- mean((data_tr$tw - ols_pred)^2)
print(MSE)

#With the large MSE output, it is clear that there is still room for improvement,
# but this value represents the lowest MS(P)E and thus the most fitting model

#Comparing all the MSPE values, we see that "mspe_ridge_selected" holds the lowest MSPE,
#meaning that selecting features using stepwise and regularization using ridge afterwards
# leads to the most accurate predictive results.





