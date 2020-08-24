rm(list=ls())
gc()
library(plyr)
library(data.table)
library(dummies)
library(VIM)
library(xgboost)
#library(h2o)
library(ROCR)
library(dplyr)
library(data.table)
library(pROC)
library(sp)
library(raster)
library(geojsonio)
library(mapview)

############Lectura de la subdivisión creada anterior###############

Poly <- geojson_read("barrios_catastrales/bogota_subdivi_100.geojson", what = "sp")


mapview(Poly)

###################################################################


M_X<-readRDS("Matriz_X_modelo.RDS")
M_X_copy<-M_X



####################### Alinear coordenadas ###########

coordinates(M_X_copy) <- ~ LONG + LAT
proj4string(M_X_copy) <- proj4string(Poly)

##############Generar tabla con los puntos encontrados ###############
M_X$ID_Poly<-over(M_X_copy, Poly)
M_X
M_X<-M_X[!is.na(ID_Poly)]





M_X[,IDX:=paste0(Semana,"_",ID_Poly)]

M_X[,IDX1:=paste0(Semana,"_",ID_Poly,"_",CATEGORIA_SUPER_APP)]



M_X_IDX<-M_X[,list(Facturacion_Total_Pol=sum(Facturacion),Trx_Total_Pol=sum(TRX),Tarj_Total_Pol=sum(Tarjetas),Avg_Tarj_Pol=mean(Tarjetas),
                     Num_Comercio_Pol=uniqueN(CU)),by=c("IDX")]




M_X_IDX1<-M_X[,list(Facturacion_Total_Pol_Cat=sum(Facturacion),Trx_Total_Pol_Cat=sum(TRX),Tarj_Total_Pol_Cat=sum(Tarjetas),Avg_Tarj_Pol_Cat=mean(Tarjetas),
                     Num_Comercio_Pol_Cat=uniqueN(CU)),by=c("IDX1")]




Y<-readRDS("y_con_pol.RDS")
Y_INFO<-readRDS("y_model.RDS")
Y_INFO<-Y_INFO[,.(CU,CATEGORIA_SUPER_APP)]


Y<-merge(Y,Y_INFO,by="CU")

Y[, Y_Total := rowSums(.SD,na.rm = T), .SDcols = 8:37]
Y[,Y_Total_Target:=ifelse(Y_Total>=10,1,0)]
table(Y$Y_Total_Target)

Y<-Y[,.(CU,Semana,LONG,LAT,CATEGORIA_SUPER_APP,Y_Total_Target)]

Y_copy<-Y



####################### Alinear coordenadas ###########

coordinates(Y_copy) <- ~ LONG + LAT
proj4string(Y_copy) <- proj4string(Poly)

##############Generar tabla con los puntos encontrados ###############
Y$ID_Poly<-over(Y_copy, Poly)
Y
Y<-Y[!is.na(ID_Poly)]



Y[,IDX:=paste0(Semana,"_",ID_Poly)]
Y[,IDX1:=paste0(Semana,"_",ID_Poly,"_",CATEGORIA_SUPER_APP)]
Y
#############

####################################################

combin <- dummy.data.frame(Y, names = c("CATEGORIA_SUPER_APP"), sep = "_")%>%setDT()
combin

combin<-merge(M_X_IDX,combin,by="IDX",all.y = T)



BASE_MODELO<-combin[,.(IDX,IDX1,CU,CATEGORIA_SUPER_APP_SUPERMERCADOS,CATEGORIA_SUPER_APP_RESTAURANTES,
                       CATEGORIA_SUPER_APP_DROGUERIAS,CATEGORIA_SUPER_APP_HOGAR,CATEGORIA_SUPER_APP_MASCOTAS,CATEGORIA_SUPER_APP_EVENTOS,
                       Facturacion_Total_Pol, Trx_Total_Pol, Tarj_Total_Pol,Avg_Tarj_Pol, Num_Comercio_Pol,
                       Y_Total_Target)]


FINAL<-merge(M_X_IDX1,BASE_MODELO,by="IDX1",all.y = T)

FINAL_F<-FINAL[,c("IDX","IDX1","CU"):=NULL]

#################################
set.seed(56897)
inx<-sample(seq(1,2),size=nrow(FINAL_F),replace=T,prob=c(0.7,0.3))
# c.train <- combin[1:nrow(train),]
c.train1 <- FINAL_F[inx==1,]

# c.test <- combin[-(1:nrow(train)),]
c.test1 <- FINAL_F[inx==2,]


######9.definición variable objetivo (malos)#####
y_train<-as.matrix(c.train1[,Y_Total_Target])
y_test<-as.matrix(c.test1[,Y_Total_Target])
y_all<-as.matrix(FINAL_F[,Y_Total_Target])


c.train1 <- c.train1[,"Y_Total_Target":=NULL]
c.test1 <- c.test1[,"Y_Total_Target":=NULL]
c.all <- FINAL_F[,"Y_Total_Target":=NULL]



train.matrix = as.matrix(c.train1)
mode(train.matrix) = "numeric"
test.matrix = as.matrix(c.test1)
mode(test.matrix) = "numeric"
all.matrix = as.matrix(c.all)
mode(all.matrix) = "numeric"




##################################

dtrain <- xgb.DMatrix(train.matrix, label = y_train)
dtest <- xgb.DMatrix(test.matrix, label = y_test)
watchlist <- list(eval = dtest, train = dtrain)


set.seed(2118)
param <- list("objective" = "binary:logistic",    # multiclass classification
              "eval_metric" = "auc",    # evaluation metric
              "nthread" = 4,   # number of threads to be used
              "max_depth" =5 ,    # maximum depth of tree
              "eta" = 0.01,    # step size shrinkage
              "gamma" = 0,    # minimum loss reduction
              "subsample" =0.81,    # part of data instances to grow tree
              "colsample_bytree" = 0.9,  # subsample ratio of columns when constructing each tree
              "min_child_weight" = 5,  # minimum sum of instance weight needed in a child
              "scale_pos_weight" = sum(BASE_MODELO$Y_Total_Target==0)/sum(BASE_MODELO$Y_Total_Target==1)
)

bst <- xgb.train(param, dtrain, nrounds = 100, watchlist)

matplot(bst$evaluation_log[,2:3],type = "l")
print(max(bst$evaluation_log$eval_auc))


xgb.importance(model = bst)

#bst <- xgb.train(param, dtrain, nrounds = 14, watchlist)


#########RESULTADO##########
library(ROCR)
pred <- predict(bst, train.matrix)  
pred.1 <- ROCR::prediction(pred,y_train)
perf.1 <- performance(pred.1,"tpr", "fpr")
plot(perf.1, colorize=TRUE)
plot(perf.1, colorize=TRUE)
abline(a=0, b= 1)
m1.auc <- performance(pred.1, "auc")
m1.auc@y.values[[1]]

2*(m1.auc@y.values[[1]]-0.5)

plot.roc(y_train, pred,
         
         main="Confidence interval of a threshold", percent=TRUE,
         
         ci=TRUE, of="thresholds", # compute AUC (of threshold)
         print.auc=TRUE,
         thresholds="best", # select the (best) threshold
         
         print.thres="best") # also highlight this threshold on the plot
#########RESULTADO##########
library(ROCR)
pred <- predict(bst, test.matrix)  
hist(pred)
pred.1 <- ROCR::prediction(pred,y_test)
perf.1 <- performance(pred.1,"tpr", "fpr")
plot(perf.1, colorize=TRUE)
abline(a=0, b= 1)
m1.auc <- performance(pred.1, "auc")
m1.auc@y.values[[1]]

2*(m1.auc@y.values[[1]]-0.5)


plot.roc(y_test, pred,
         
         main="Confidence interval of a threshold", percent=TRUE,
         
         ci=TRUE, of="thresholds", # compute AUC (of threshold)
         
         thresholds="best", # select the (best) threshold
         print.auc=TRUE,
         
         print.thres="best") # also highlight this threshold on the plot


hist(pred)


table(y_test,pred<0.5)

#################
library(ROCR)
pred <- predict(bst, all.matrix)  
pred.1 <- ROCR::prediction(pred,y_all)
perf.1 <- performance(pred.1,"tpr", "fpr")
plot(perf.1, colorize=TRUE)
m1.auc <- performance(pred.1, "auc")
m1.auc@y.values[[1]]
2*(m1.auc@y.values[[1]]-0.5)

hist(pred)

table(y_test,pred<0.47)

write.csv(data.frame(TARGET=y_all,PRED=pred),file="ALL.csv")

library(pROC)

data(aSAH)



plot.roc(y_test, pred,
         
         main="Confidence interval of a threshold", percent=TRUE,
         
         ci=TRUE, of="thresholds", # compute AUC (of threshold)
         
         thresholds="best", # select the (best) threshold
         
         print.thres="best") # also highlight this threshold on the plot




library(ggplot2)
DD<-data.frame(test.matrix,y_test,pred=predict(bst, test.matrix) )

ggplot(DD, aes(pred, fill = as.factor(y_test))) +
  geom_density(position = "stack")


max(attr(perf.1,'y.values')[[1]]-attr(perf.1,'x.values')[[1]])
