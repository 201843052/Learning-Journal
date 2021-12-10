library(MASS)
library(glasso)

setClass(Class="Performance",
         representation(
           metric1="numeric",
           metric2="numeric",
           metric3="numeric"
         )
)
AUC = function(FP, TP){
  sorted.FP = sort(FP)
  sorted.TP = sort(TP)
  a = head(sorted.TP, -1)
  b = tail(sorted.TP, -1)
  h = diff(sorted.FP)
  s = sum((a + b) * h) / 2
  return(s)
}
GlassoGridSearchCV <- function(X_train,grid,k=5){
  
  fold = sample(rep(1:k,length = nrow(X_train)))
  print(paste("Grid Size:",length(grid),sep = " "))
  
  nllCV = rep(0,length(grid))
  
  for(i in (1:length(grid))){
    temp = 0
    for(j in (1:k)){
      train = X_train[fold!=j,]
      val = X_train[fold==j,]
      s = cov(train)
      n = nrow(train)
      # fit glasso using training set
      theta_hat = glasso(s,rho = grid[i],nobs=n)
      # calculate negative log-likelihood using validation set
      nll = -determinant(theta_hat$wi,logarithm = T)$modulus+sum(diag(cov(val)%*%theta_hat$wi))
      temp = temp + nll
    }
    # Average negative log-likelihood across k fold
    nllCV[i] = temp/k
  }
  # metric1 collects nll for all lambda, metric2 collects the sd of nll
  return(new("Performance",metric1=nllCV,metric2=sd(nllCV),metric3=0))
  
}
Glasso <- function(X,lambda){
  
  s = cov(X)
  n = nrow(X)
  
  theta.hat = glasso(s,rho = lambda,nobs=n)$wi
  E3 = theta.hat[upper.tri(theta.hat)]!=0
  E3 = as.integer(E3)
  return(E3)
}
GenerateSparseMatrix <- function(n,p,sparsity = 0.1){
  B = matrix(0L,nrow=p,ncol=p)
  while(sum(B)==0){
    offdiag = runif((p**2-p)/2)
    for(i in 1:p){
      B[i,i]=0
      for(j in 1:p){
        if (i!=j){
          if(offdiag[i]<=sparsity){
            B[i,j]=0.5
            B[j,i]=0.5
          } else {
            B[i,j]=0
            B[j,i]=0
          }
        }
      }
    }
  }
  
  eigen(B)$values
  delta = -min(eigen(B)$values)+0.01
  diag(B) = delta
  
  return(B)
}
GlassoGridSearch <- function(mnorm,grid,trueE){
  
  s = cov(mnorm)
  n = nrow(mnorm)
  
  NP = sum(trueE)
  NN = length(trueE)-NP
  
  print(paste("Grid Size:",length(grid),sep = " "))
  
  tpr.E3 = 0
  fpr.E3 = 0
  me.E3 = 0
  for ( i in (1:length(grid))){
    print(i)
    theta.hat = glasso(s,rho = grid[i],nobs=n)$wi
    temp.E3 = theta.hat[upper.tri(theta.hat)]!=0
    temp.E3 = as.integer(temp.E3)
    tp = sum((temp.E3==1)&(trueE==1))
    tn = sum((temp.E3==0)&(trueE==0))
    fp = sum((temp.E3==1)&(trueE==0))
    fn = sum((temp.E3==0)&(trueE==1))
    me.E3[i] = (fn+fp)/(choose(p,2))
    tpr.E3[i] = tp/NP
    fpr.E3[i] = fp/NN
  }
  return(new("Performance",metric1=tpr.E3,metric2=fpr.E3,metric3=me.E3))
}
NodeWiseGridSearchCV <- function(X_train,grid,k = 5){
  fold = sample(rep(1:k,length = nrow(X_train)))
  
  print(paste("Grid Size:",length(grid),sep = " "))
  
  lambda.mse = 0
  
  for(i in (1:length(grid))){
    temp.mse = 0
    for(j in (1:k)){
      train = X_train[fold!=j,]
      val = X_train[fold==j,]
      # fit model using train, and predict, which compares with val for each node-wise
      for(t in (1:p)){
        fit.lasso = glmnet(train[,-t],train[,t],alpha=1,lambda = grid[i])
        df.lasso = length(fit.lasso[["beta"]]@i)
        prediction = predict(fit.lasso,lambda = grid[i], newx=val[,-t])
        rss = sum((prediction - val[,t])^2)
        mse = rss /nrow(val)
        temp.mse = temp.mse + mse
      }
    }
    lambda.mse[i] = temp.mse/k
  }
  return(new("Performance",metric1=0,metric2=lambda.mse,metric3=0))
}
NodeWiseGridSearchShuffle <- function(X_train,grid){
  print(paste("Grid Size:",length(grid),sep = " "))
  lambda.mse = 0
  X_train = X_train[sample(1:nrow(X_train)), ]
  # shuffle 50% 50% split
  train = X_train[(1:n/2),] 
  val = X_train[((n/2)+1):n,]
  for(i in (1:length(grid))){
    temp.mse = 0
    
    for(t in (1:ncol(X_train))){
      fit.lasso = glmnet(train[,-t],train[,t],alpha=1,lambda = grid[i])
      prediction = predict(fit.lasso,lambda = grid[i],newx = val[,-t])
      rss = sum((prediction - val[,t])^2)
      mse = rss /nrow(val)
      temp.mse = temp.mse + mse
    }
    lambda.mse[i] = temp.mse/ncol(X_train)
  }
  return(new("Performance",metric1=0,metric2=lambda.mse,metric3=0))
}
NodeWiseGridSearchTrueE <- function(mnorm,grid,trueE,type){
  print(paste("Grid Size:",length(grid),sep = " "))
  pool = c(1:ncol(mnorm))
  NP = sum(trueE)
  NN = length(trueE)-NP
  
  if(NP==0){
    print("Empty True Edge")
    return()
  }
  
  tpr.type = 0
  fpr.type = 0
  me.type = 0
  for (i in 1:length(grid)){
    print(i)
    
    betas.nocv = matrix(0L,nrow=p,ncol=p)
    for (j in 1:p){
      fit.lasso = glmnet(mnorm[,-j],mnorm[,j],alpha=1,lambda = grid[i])
      temp = coef(fit.lasso)@i
      if(length(temp[-1])==1){
        if(coef(fit.lasso)[2]==0){
          next
        }
      }
      betas.nocv[j,pool[-j][temp[-1]]]=1
    }
    if(type==1){
      temp = betas.nocv[upper.tri(betas.nocv)]==1 & betas.nocv[lower.tri(betas.nocv)]==1
      temp = as.integer(temp)
    } else if(type==2){
      temp = betas.nocv[upper.tri(betas.nocv)]==1 | betas.nocv[lower.tri(betas.nocv)]==1
      temp = as.integer(temp)
    } else{
      print("Choose type=1 for E1, type=2 for E2.")
      return()
    }
    tp = sum((temp==1)&(trueE==1))
    tn = sum((temp==0)&(trueE==0))
    fp = sum((temp==1)&(trueE==0))
    fn = sum((temp==0)&(trueE==1))
    
    me.type[i] = (fn+fp)/(choose(p,2))
    tpr.type[i] = tp/NP
    fpr.type[i] = fp/NN
    
  }
  
  return(new("Performance",metric1 = tpr.type,metric2= fpr.type,metric3 = me.type))
  
}
NodeWise<-function(mnorm,lambda,type){
  pool = c(1:p)
  betas = matrix(0L,nrow=p,ncol=p)
  
  for (j in 1:p){
    fit.lasso = glmnet(mnorm[,-j],mnorm[,j],alpha=1,lambda = lambda)
    temp = coef(fit.lasso)@i
    if(length(temp[-1])==1){
      if(coef(fit.lasso)[2]==0){
        next
      }
    }
    betas[j,pool[-j][temp[-1]]]=1
  }
  
  if(type==1){
    E = (betas[upper.tri(betas)]==1) & (betas[lower.tri(betas)]==1)
    E = as.integer(E)
  } else if(type==2){
    E = (betas[upper.tri(betas)]==1) | (betas[lower.tri(betas)]==1)
    E = as.integer(E)
  }
  
  return(E)
}
p = 20
n = 15
sparsity = 0.1
B = GenerateSparseMatrix(n,p,sparsity)
theta = cov2cor(B)

# testing set
trueE = theta[upper.tri(theta)]!=0
trueE = as.integer(trueE)

# Preparing for training set
sigma = solve(theta)
mean = rep(0,p)
mnorm = mvrnorm(n,mean,sigma)

# Get a sense of the model by exhaustive search
lambdas = c(seq(0.0001,0.5,0.001),seq(0.5,1,0.01),seq(1,50,0.1)) # works for p20n1000, p50n1000, p50n40
sneakypeak = GlassoGridSearch(mnorm,lambdas,trueE)
auc.E3 = AUC(sneakypeak@metric2,sneakypeak@metric1)
plot(sneakypeak@metric2,sneakypeak@metric1,
     main=paste("ROC E3",auc.E3,"p",p,"n",n,sep = " "),
     xlim=c(0,1),ylim=c(0,1),type="l",
     xlab="fpr",ylab="tpr")
points(sneakypeak@metric2,sneakypeak@metric1,cex=0.5)
abline(0,1)

lambdas = c(seq(0.0001,0.5,0.001),seq(0.5,50,0.1)) # works for p20n1000, p50n1000, p50n40
sneakypeak.E1 = NodeWiseGridSearchTrueE(mnorm,lambdas,trueE,1)
auc.E1 = AUC(sneakypeak.E1@metric2,sneakypeak.E1@metric1)
plot(sneakypeak.E1@metric2,sneakypeak.E1@metric1,
     main=paste("ROC E1",auc.E1,"p",p,"n",n,sep = " "),
     xlim=c(0,1),ylim=c(0,1),type="l",
     xlab="fpr",ylab="tpr")
points(sneakypeak.E1@metric2,sneakypeak.E1@metric1,cex=0.5)
abline(0,1)
sum(trueE)

lambdas = c(seq(0.0001,0.5,0.001),seq(0.5,100,0.1))
sneakypeak.E2 = NodeWiseGridSearchTrueE(mnorm,lambdas,trueE,2)
auc.E2 = AUC(sneakypeak.E2@metric2,sneakypeak.E2@metric1)
plot(sneakypeak.E2@metric2,sneakypeak.E2@metric1,
     main=paste("ROC E2",auc.E2,"p",p,"n",n,sep = " "),
     xlim=c(0,1),ylim=c(0,1),type="l",
     xlab="fpr",ylab="tpr")
points(sneakypeak.E2@metric2,sneakypeak.E2@metric1,cex=0.5)
abline(0,1)
sum(trueE)


# Find constant to multiply theoretical_lambda
constant.list.GL = 0
constant.list.NW = 0
lambdas = c(seq(0.0001,0.5,0.0001),seq(0.5,1,0.01),seq(1,50,0.1)) # maybe make finer and wider
theoretical_lambda = sqrt(log(p)/n)

for(num in 1:15){
  print(num)
  Glasso.Performance = GlassoGridSearchCV(mnorm,lambdas,5)
  lambdamin = lambdas[which.min(Glasso.Performance@metric1)]
  # Here we use the "one standard deviation rule"
  # constant.list[num] = (lambdamin+Glasso.Performance@metric2)/(theoretical_lambda)
  constant.list.GL[num] = (lambdamin)/(theoretical_lambda)
  
  NodeWise.Performance.mse = NodeWiseGridSearchCV(mnorm,lambdas,5)
  # NodeWise.Performance.BIC = NodeWiseGridSearchBIC(mnorm,lambdas)
  # Best lambda using lowest BIC, change metric1 to metric2 to use MSE
  # constant.list.bic[num] = lambdas[which.min(NodeWise.Performance.BIC@metric1)]/(theoretical_lambda)
  constant.list.NW[num] = lambdas[which.min(NodeWise.Performance.mse@metric2)]/(theoretical_lambda)
}
boxplot(constant.list.GL,main=paste("Glasso Constant p=",p," n=",n,sep=""))
boxplot(constant.list.NW,main=paste("Node-Wise Constant p=",p," n=",n,sep=""))
constant.NW = mean(constant.list.NW)
span.NW = sd(constant.list.NW)*(theoretical_lambda)
constant.GL = mean(constant.list.GL)
span.GL = sd(constant.list.GL)*(theoretical_lambda)

# Set lambda grid to be centered around constant*theoretical, and symmetrically spread by 1 sd = theoretical^2*sd(span)
# Preparing for 50 iterations
if (constant.GL*theoretical_lambda-1.96*span.GL < 0){
  lowerbound.GL = 0.0001
} else {
  lowerbound.GL = constant.GL*theoretical_lambda-1.96*span.GL
}
refined_grid.GL = c(seq(lowerbound.GL,constant.GL*theoretical_lambda+1.96*span.GL,0.001))

if (constant.NW*theoretical_lambda-1.96*span.NW < 0){
  lowerbound.NW = 0.0001
} else {
  lowerbound.NW = constant.NW*theoretical_lambda-1.96*span.NW
}
refined_grid.NW = c(seq(lowerbound.NW,constant.NW*theoretical_lambda+1.96*span.NW,0.0001))
me.E1 = 0
me.E2 = 0
tpr.E1 = 0
tpr.E2 = 0
fpr.E1 = 0
fpr.E2 = 0
precesion.E1 = 0
precesion.E2 = 0
me.E3 = 0
tpr.E3 = 0
fpr.E3 = 0
precesion.E3 = 0
for(fiftyidx in  1:50){
  B = GenerateSparseMatrix(n,p)
  theta = cov2cor(B)
  # testing set
  trueE = theta[upper.tri(theta)]!=0
  trueE = as.integer(trueE)
  
  # Preparing for training set
  sigma = solve(theta)
  mean = rep(0,p)
  mnorm = mvrnorm(n,mean,sigma)
  
  Glasso.Performance = GlassoGridSearchCV(mnorm,refined_grid.GL,5)
  E3.min.lambda = refined_grid.GL[which.min(Glasso.Performance@metric1)]
  # again we use "one sd rule"
  # E3.min.lambda = E3.min.lambda + (Glasso.Performance@metric2*theoretical_lambda)
  E3 = Glasso(mnorm,E3.min.lambda)
  
  me.E3[fiftyidx] = sum(abs(E3-trueE))/length(E3)
  tp = sum((E3==1)&(trueE==1))
  tn = sum((E3==0)&(trueE==0))
  fp = sum((E3==1)&(trueE==0))
  fn = sum((E3==0)&(trueE==1))
  tpr.E3[fiftyidx] = tp/(tp+fn)
  fpr.E3[fiftyidx] = fp/(fp+tn)
  # if somehow we decide that predicting 0 is not so important
  precesion.E3[fiftyidx] = tp/(tp+fp)
  
  
  NodeWise.Performance = NodeWiseGridSearchCV(mnorm,refined_grid.NW,5)
  min.lambda = refined_grid.NW[which.min(NodeWise.Performance@metric2)]
  # again we use "one sd rule"
  # E3.min.lambda = E3.min.lambda + (Glasso.Performance@metric2*theoretical_lambda)
  E1 = NodeWise(mnorm,min.lambda,1)
  E2 = NodeWise(mnorm,min.lambda,2)
  
  me.E1[fiftyidx] = sum(abs(E1-trueE))/length(E1)
  me.E2[fiftyidx] = sum(abs(E2-trueE))/length(E2)
  tp.E1 = sum((E1==1)&(trueE==1))
  tn.E1 = sum((E1==0)&(trueE==0))
  fp.E1 = sum((E1==1)&(trueE==0))
  fn.E1 = sum((E1==0)&(trueE==1))
  tp.E2 = sum((E2==1)&(trueE==1))
  tn.E2 = sum((E2==0)&(trueE==0))
  fp.E2 = sum((E2==1)&(trueE==0))
  fn.E2 = sum((E2==0)&(trueE==1))
  tpr.E1[fiftyidx] = tp.E1/(tp.E1+fn.E1)
  fpr.E1[fiftyidx] = fp.E1/(fp.E1+tn.E1)
  tpr.E2[fiftyidx] = tp.E2/(tp.E2+fn.E2)
  fpr.E2[fiftyidx] = fp.E2/(fp.E2+tn.E2)
  # if somehow we decide that predicting 0 is not so important
  precesion.E1[fiftyidx] = tp.E1/(tp.E1+fp.E1)
  precesion.E2[fiftyidx] = tp.E2/(tp.E2+fp.E2)
}
par(mfrow=c(2,2))
boxplot(tpr.E3,main=paste("E3's TPR p=",p," n=",n,sep=""))
boxplot(fpr.E3,main="E3's FPR")
boxplot(precesion.E3,main="E3's Precesion")
boxplot(me.E3,main="E3's Missclass")

boxplot(tpr.E1,main=paste("E1's TPR p=",p," n=",n,sep=""))
boxplot(fpr.E1,main="E1's FPR")
boxplot(precesion.E1,main="E1's Precesion")
boxplot(me.E1,main="E1's Missclass")

boxplot(tpr.E2,main=paste("E2's TPR p=",p," n=",n,sep=""))
boxplot(fpr.E2,main="E2's FPR")
boxplot(precesion.E2,main="E2's Precesion")
boxplot(me.E2,main="E2's Missclass")
par(mfrow=c(1,1))
