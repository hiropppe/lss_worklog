
data {
  int<lower=1> M;                    // num docs
  int<lower=1> V;                    // num words
  int<lower=1> N;                    // total word instances
  int<lower=1,upper=V> W[N];         // word n
  int<lower=1> D[N];                 // doc id
  vector[V] p_v;                     // word prob
  vector[V] phi;                     // word polarity
}

parameters {
  vector[M] theta;
}

model {
  theta ~ normal(0, 1);
  for (n in 1:N)
    W[n] ~ categorical_logit(log(p_v) + phi * theta[D[n]]);
}
