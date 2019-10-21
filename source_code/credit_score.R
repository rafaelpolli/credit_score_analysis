library(ggplot2)
library(gridExtra)
library(dplyr)
library(dummies)
library(randomForest)
library(caTools)
library(caret)
library(ROCR)
library(corrplot)
library(parallel)
library(doParallel)
source("plot_utils.R") #Para criação da curva ROC

df <- read.table("credit_dataset.csv", sep = ",", header = TRUE)
View(df)

str(df)
df[,-c(3,6,14)] <- lapply(df[,-c(3,6,14)], as.factor)
str(df)
summary(df)

grid.arrange(ggplot(df, aes(x=account.balance, fill=credit.rating))+ geom_bar(position = "dodge"),
             ggplot(df, aes(x=dependents, fill=credit.rating))+ geom_bar(position = "dodge"),
             ggplot(df, aes(x=credit.duration.months, fill=credit.rating))+ geom_histogram(position = "dodge", binwidth = 5),
             ggplot(df, aes(x=credit.amount, fill=credit.rating))+ geom_histogram(position = "dodge"), 
             ncol=2)

grid.arrange(ggplot(df, aes(x=credit.purpose, fill=credit.rating))+ geom_bar(position = "dodge"),
             ggplot(df, aes(x=savings, fill=credit.rating))+ geom_bar(position = "dodge"),
             ggplot(df, aes(x=employment.duration, fill=credit.rating))+ geom_bar(position = "dodge"),
             ggplot(df, aes(x=age, fill=credit.rating))+ geom_bar(position = "dodge"), 
             ncol=2)


set.seed(123)


cluster <- makeCluster(detectCores() - 1) #Parametro para detectar o números de núcleos do processador para processamento paralelo
registerDoParallel(cluster) #Parametro para detectar o números de núcleos do processador para processamento paralelo

Base1 <- subset(df, df$credit.rating == 1)
Base0 <- subset(df, df$credit.rating == 0)

dim(Base1)
dim(Base0)

dt = sort(sample(nrow(Base1), 300))
Amostra_1 <- Base1[dt,] 
base_balanceada = rbind(Base0, Amostra_1)
table(base_balanceada$credit.rating)

nums <- unlist(lapply(df, is.numeric))
cor(df[,nums])

amostra <- sample.split(base_balanceada$credit.rating, SplitRatio = 0.70)

treino = subset(base_balanceada, amostra == TRUE)

teste = subset(base_balanceada, amostra == FALSE)

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

modelo_fs <- randomForest(credit.rating ~ .
                          ,data = treino,
                          na.action = na.roughfix)

importance <- varImp(modelo_fs, scale=FALSE)

print(importance)

train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3, search = "random", allowParallel = TRUE)

model <- train(credit.rating ~ credit.amount
               + account.balance
               + credit.duration.months
               + age
               , data = treino, method = "rf",
               metric = "Accuracy",
               tuneLength=15,
               trControl = train.control, na.action = na.roughfix)
model

summary(model)

previsoes <- data.frame(observado = teste$credit.rating,
                        previsto = predict(model, newdata = teste))

confusionMatrix(previsoes$observado, previsoes$previsto)

predictions <- prediction(as.numeric(previsoes$previsto), as.numeric(previsoes$observado))

plot.roc.curve(predictions, title.text = "Curva ROC")
plot.pr.curve(predictions, title.text = "Curva Precision/Recall")

modelo2 <- glm(credit.rating ~
                 .,
               family=binomial(link='logit'),data=treino)

summary(modelo2)

base_balanceada2 <- dummy.data.frame(base_balanceada, names = c("account.balance", "previous.credit.payment.status",
                                                                "credit.purpose", "savings", "employment.duration",
                                                                "installment.rate", "marital.status", "guarantor",
                                                                "residence.duration", "current.assets", "other.credits",
                                                                "apartment.type", "bank.credits", "occupation", "dependents",
                                                                "telephone", "foreign.worker"
))

str(base_balanceada2)

treino2 = subset(base_balanceada2, amostra == TRUE)

teste2 = subset(base_balanceada2, amostra == FALSE)

step(glm(credit.rating ~
           .,
         family=binomial(link='logit'),data=treino2), direction = "both")

modelo3 <-glm(formula = credit.rating ~ account.balance1 + account.balance2 + 
                credit.duration.months + previous.credit.payment.status1 + 
                credit.purpose1 + credit.amount + savings1 + savings2 + residence.duration2 + 
                other.credits1 + apartment.type1 + telephone1 + foreign.worker1, 
              family = binomial(link = "logit"), data = treino2)


summary(modelo3)

pred2 = predict(modelo3, teste2, type = "response")

base_final2 = cbind(teste2, pred2)

base_final2$resposta <- as.factor(ifelse(base_final2$pred>0.5, 1, 0))

confusionMatrix(base_final2$credit.rating, base_final2$resposta)

predictions <- prediction(as.numeric(pred2), as.numeric(base_final2$credit.rating))

plot.roc.curve(predictions, title.text = "Curva ROC")
plot.pr.curve(predictions, title.text = "Curva Precision/Recall")

