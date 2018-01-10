functions {
  real lifetime(real M, real[] ltc) {
    real logM = log(M);
    real logMAcc = logM;
    int nltc = size(ltc);

    real log_lt = ltc[1]; /* Constant term */
    for (i in 2:nltc) {
      log_lt = log_lt + ltc[i]*logMAcc;

      logMAcc = logM*logMAcc; /* logMAcc = logM^i */
    }

    return exp(log_lt);
  }

  real[] log_dPdMdt(real[] M, real[] t, real alpha, real mu, real sigma, real MMin, real tmax) {
    real logdPdMdt[size(M)];

    real normM = MMin/(alpha-1.0);
    real normt = normal_cdf(tmax, mu, sigma) - normal_cdf(0.0, mu, sigma);
    
    for (i in 1:size(M)) {
      logdPdMdt[i] = -alpha*log(M[i]/MMin) + normal_lpdf(t[i] | mu, sigma) - log(normM) - log(normt);
    }

    return logdPdMdt;
  }

  real lum_time_limit(real M, real tau0, real beta, real MComplete) {
    return tau0*(1-M/MComplete)^beta;
  }
}

data {
  int nltc; /* Number of lifetime fit coefficients. */
  real ltc[nltc]; /* Fit coefficients for stellar lifetime versus mass. */

  real MMin; /* Minimum mass considered. */
  real MComplete; /* Mass above which survey is complete. */
  real tmax; /* Largest age considered. */

  int nobs;
  real Mobs[nobs];
  real sigma_logM[nobs];
  real ageobs[nobs];
  real sigma_logage[nobs];

  int nundet_max; /* Maximum permitted number of un-detected sources
		     permitted. */
}

parameters {
  real<lower=0> L; /* Count / number of stars formed (detected + undetected) */
  real<lower=0> Lnp; /* Count of non-physical stars (from the not
			undetected subset of the possibly undetected
			systems) .*/

  real<lower=1> alpha; /* Power-law index on IMF: dN/dM ~ M^-alpha */
  
  real<lower=0,upper=tmax> mu_t; /* Mean of Gaussian-shape for SFR */
  real<lower=0,upper=tmax> sigma_t; /* S.D. of Gaussian-shape SFR. */
  
  real Mtrue[nobs];
  real<lower=0, upper=1> fttrue[nobs]; /* True age is fttrue*lifetime(Mtrue) */

  /* Parameters describing luminosity limit. */
  real<lower=0> beta;
  real<lower=0> tau0;
  
  /* For the (possibly) undetected systems.  Because Stan does not
     have the possibility of combining ordered and bounded vectors, we
     will have to implement the Jacobian manually. */
  positive_ordered[nundet_max] logMoMMin_undet;
  real<lower=0, upper=tmax> agetrue_undet[nundet_max];
}

transformed parameters {
  real agetrue[nobs];
  real Mtrue_undet[nobs];

  for (i in 1:nobs) {
    agetrue[i] = fttrue[i]*lifetime(Mtrue[i], ltc);
  }

  for (i in 1:nundet_max) {
    Mtrue_undet[i] = MMin*exp(logMoMMin_undet[i]);
  }
}

model {
  /* Priors */
  L ~ normal(0.0, nobs+nundet_max);
  Lnp ~ normal(0.0, nundet_max);

  alpha ~ cauchy(0.0, 1.0);

  /* mu_t flat between limits */
  sigma_t ~ cauchy(0.0, tmax/2.0);
  
  beta ~ cauchy(0.0, 1.0);
  tau0 ~ cauchy(0.0, tmax/10.0);

  /* Observed systems */
  /* Likelihood terms */
  Mobs ~ lognormal(log(Mtrue), sigma_logM);
  ageobs ~ lognormal(log(agetrue), sigma_logage);
  /* Hierarchical (Population) Prior */
  target += nobs*log(L);
  target += sum(log_dPdMdt(Mtrue, agetrue, alpha, mu_t, sigma_t, MMin, tmax));
  /* Jacobian for age.  Because the sampling variable is f =
     age/lifetime, we need to multiply if d(age)/df = 1/(df/d(age)) =
     lifetime = age/f for each system */
  target += sum(to_vector(log(agetrue)) - to_vector(log(fttrue)));

  /* (Possibly) Un-observed systems.  One of these systems can be
     "physical" only if:

     * M < MComplete and age < tau0*(1-M/MComplete)^beta
     * age > lifetime(M)

     Systems that *could* be physcial are counted in a mixture model
     of L and Lnp; non-physical systems are counted only in Lnp.
  */
  for (i in 1:nundet_max) {
    /* Hierarchical (Population) prior */
    target += sum(log_dPdMdt(Mtrue_undet, agetrue_undet, alpha, mu_t, sigma_t, MMin, tmax));
    /* Jacobian. Sampling parameter is x = log(M/MMin), so we need to
       multiply by dM/dx = 1/(dx/dM) = M for each undet object. */
    target += sum(log(Mtrue_undet));
    
    if (Mtrue_undet[i] < MComplete) {
      if (agetrue_undet[i] < lum_time_limit(Mtrue_undet[i], tau0, beta, MComplete)) {
	target += log_sum_exp(log(L), log(Lnp));
      } else if (agetrue_undet[i] > lifetime(Mtrue_undet[i], ltc)) {
	target += log_sum_exp(log(L), log(Lnp));
      } else {
	target += log(Lnp);
      }
    } else {
      if (agetrue_undet[i] > lifetime(Mtrue_undet[i], ltc)) {
	target += log_sum_exp(log(L), log(Lnp));
      } else {
	target += log(Lnp);
      }
    }
  }

  target += -L - Lnp; /* Poisson normalisation. */
}
