
##NOTE: wordmat must be a matrix (i.e. wfm object), not a data frame

### Function currently fixes identifying constraints: beta[1] and alpha[1] don't update, are fixed at whatever is input as start
wordfish.het <- function(wordmat, Nsamples, beta.start, gamma.start, alpha.start, psi.start){

slice<-function(f,N,init,w,p,burnin,thin,log=FALSE,...) {

samples<-numeric(N)
samples[1]<-init		#startval
prev.density<-f(init,...)	#initial density

for(i in 2:N){

if(log) {
y<-prev.density-rexp(1)
} else{
y<-runif(1,min=0,max=prev.density)
}
L <- samples[i-1] - w*runif(1)
R <- L + w
K <- p

#doubling procedure -- looks weird because we avoid double conditional ORs
flag<-FALSE
if(y < f(L,...)) {
flag<-TRUE
} else {
if(y <f(R,...)) flag<-TRUE
}

while(K>0 & flag) {
if(runif(1)<0.5){ 
L <- 2*L - R
if(y >= f(L,...)) flag<-FALSE
} else {
R <- 2*R - L
if(y >= f(R,...)) flag<-FALSE
}
K <- K-1
}

#Shrinkage procedure
L.bar <- L
R.bar <- R

repeat{
x1 <- L.bar + runif(1)*(R.bar-L.bar)
prev.density <- f(x1,...)	#saves previous density for use above
if(prev.density>y) break
if(samples[i-1] > x1) {
L.bar <- x1
} else{
R.bar <- x1
}
} #end repeat
samples[i] <- x1
}

#Postprocessing: Burnin and Thinning
samples<-samples[-(1:burnin)]
samples<-samples[seq(from=1,to=length(samples),by=thin)]
return(samples)

}#end function

##Basic cell log-likelihood
wc.llik <- function(wc, alpha.sample, beta.sample, psi.sample,
gamma.sample, r.sample){
	   if(r.sample<0.5) return(-9999999999999)
	   mu <- exp(alpha.sample + psi.sample + beta.sample*gamma.sample)
	   return(dnbinom(wc, mu=mu, size=r.sample, log=TRUE))
}

##Updates alpha, gamma, and r
doc.llik <- function(wc, alpha.sample, beta.vec, psi.vec, gamma.sample, r.sample){
	   if(r.sample<0.5) return(-9999999999999)
	   loglike <- 0
	   for(i in 1:length(wc)) loglike <- loglike +  wc.llik(wc[i], alpha.sample, beta.vec[i], psi.vec[i], gamma.sample, r.sample)
	   return(loglike)
}

##Updates beta and psi
word.llik <- function(wc, alpha.vec, beta.sample, psi.sample, gamma.vec, r.vec){
	   loglike <- 0
	   for(i in 1:length(wc)) loglike <- loglike +  wc.llik(wc[i], alpha.vec[i], beta.sample, psi.sample, gamma.vec[i], r.vec[i])
	   return(loglike)
}


samples <- Nsamples
J <- nrow(wordmat)
P <- ncol(wordmat)

##Initialize containers
alpha.samples <- matrix(NA, nrow=samples, ncol=P)
gamma.samples <- matrix(NA, nrow=samples, ncol=P)
r.samples <- matrix(NA, nrow=samples, ncol=P)
psi.samples <- matrix(NA, nrow=samples, ncol=J)
beta.samples <- matrix(NA, nrow=samples, ncol=J)

##Start values
beta.samples[1,] <- beta.start
gamma.samples[1,] <- gamma.start
alpha.samples[1,] <- alpha.start
psi.samples[1,] <- psi.start
r.samples[1,]  <- rep(30, P)

##Identifying Constraints
gamma.unconstrained <- 1:P
alpha.unconstrained <- 2:P
r.unconstrained <- 1:P
psi.unconstrained <- 1:J
beta.unconstrained <- 2:J
beta.samples[,1] <- rep(beta.samples[1,1], samples)
alpha.samples[,1] <- rep(alpha.samples[1,1], samples)

###################
### Sampling  ####
###################
#applys not used to make it more C-like
for(i in 2:samples){

##Update gamma
for(j in gamma.unconstrained) {
gamma.samples[i,j]<-slice(doc.llik,N=2,init=gamma.samples[i-1,j],w=0.5,p=3,burnin=0,thin=1,log=TRUE,
wc=wordmat[,j], alpha.sample=alpha.samples[i-1,j], r.sample=r.samples[i-1,j],
psi.vec=psi.samples[i-1,], beta.vec=beta.samples[i-1,]) 
}

##Update alpha
for(j in alpha.unconstrained) {
alpha.samples[i,j]<-slice(doc.llik,N=2,init=alpha.samples[i-1,j],w=0.5,p=3,burnin=0,thin=1,log=TRUE,
wc=wordmat[,j], gamma.sample=gamma.samples[i,j], r.sample=r.samples[i-1,j],
psi.vec=psi.samples[i-1,], beta.vec=beta.samples[i-1,]) 
}

##Update R
for(j in r.unconstrained) {
r.samples[i,j]<-slice(doc.llik,N=2,init=r.samples[i-1,j],w=0.5,p=3,burnin=0,thin=1,log=TRUE,
wc=wordmat[,j], alpha.sample=alpha.samples[i,j], gamma.sample=gamma.samples[i,j],
psi.vec=psi.samples[i-1,], beta.vec=beta.samples[i-1,]) 
}

##Update Beta
for(j in beta.unconstrained) {
beta.samples[i,j]<-slice(word.llik,N=2,init=beta.samples[i-1,j],w=0.5,p=3,burnin=0,thin=1,log=TRUE,
wc=wordmat[j,], alpha.vec=alpha.samples[i,], r.vec=r.samples[i,],
psi.sample=psi.samples[i-1,j], gamma.vec=gamma.samples[i,]) 
}

#Update Psi
for(j in psi.unconstrained) {
psi.samples[i,j]<-slice(word.llik,N=2,init=psi.samples[i-1,j],w=0.5,p=3,burnin=0,thin=1,log=TRUE,
wc=wordmat[j,], alpha.vec=alpha.samples[i,], r.vec=r.samples[i,],
beta.sample=beta.samples[i,j], gamma.vec=gamma.samples[i,]) 
}

#print verbose iteration
if(i %% 100 == 0) cat("\t\tIteration =", i, "complete...\n")
flush.console()

} #end for(i in 2:samples)

result <- list(Nwords=J, Nparties=P, beta.samples=beta.samples,
	gamma.samples=gamma.samples, alpha.samples=alpha.samples,
	psi.samples=psi.samples, r.samples=r.samples)
return(result)

}#end function
