functions {
  /* Returns the lifetime on the main sequence of a star of mass M.
     ltc are the fitted coefficients of the power series in log(M) for
     log(lifetime/Myr). */
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

  /* The log of the normalised density of stars in M-age space (MMin <
     M < infinity and 0 < age < tmax). */
  real[] log_dPdMdt(real[] M, real[] t, real alpha, real mu, real sigma, real MMin, real tmax) {
    real logdPdMdt[size(M)];

    real normM = MMin/(alpha-1.0);
    real normt = normal_cdf(tmax, mu, sigma) - normal_cdf(0.0, mu, sigma);
    
    for (i in 1:size(M)) {
      logdPdMdt[i] = -alpha*log(M[i]/MMin) + normal_lpdf(t[i] | mu, sigma) - log(normM) - log(normt);
    }

    return logdPdMdt;
  }

  /* This is a hack.  Stan doesn't have a quadrature code; instead it
     has an ODE solver.  ODE solvers can ("trivially") do quadrature.
     If you want to evaluate an integral 

     I = \int_A^B f(x) dx

     then you can just integrate the ODE

     dI/dx = f(x)

     with initial condition I(A) = 0 to x = B, and I = I(B).  That's
     what we're doing here.

     The integral in question is the integral of dP/dMd(age) over the
     *observable* region in parameter space.  The integral over age at
     fixed M is just a difference of normal_cdf's since the formation
     rate is a Gaussian.  The remaining integral over M must be done
     numerically.  This is the integrand:

     dI/dM = (alpha-1)/MMin (M/MMin)^alpha <normal CDFs>
*/
  real[] dfdM(real M, real[] state, real[] theta, real[] x_r, int[] x_i) {
    real alpha = theta[1];
    real mu = theta[2];
    real sigma = theta[3];

    int nltc = x_i[1];

    real ltc[nltc];
    real lt;
    real tmin;

    real MMin = x_r[1];
    real tmax = x_r[2];

    real normM = MMin / (alpha - 1.0);
    real It;

    real dIdM[1];

    for (i in 1:nltc) {
      ltc[i] = x_r[2+i];
    }

    lt = lifetime(M, ltc);

    It = (normal_cdf(lt, mu, sigma) - normal_cdf(0.0, mu, sigma))/(normal_cdf(tmax, mu, sigma) - normal_cdf(0.0, mu, sigma));

    dIdM[1] = (M/MMin)^(-alpha)/normM*It;

    return dIdM;
  }
}

data {
  int nltc; /* Number of lifetime fit coefficients. */
  real ltc[nltc]; /* Fit coefficients for stellar lifetime versus mass. */

  real MMin; /* Minimum mass considered. */
  real tmax; /* Largest age considered. */

  int nobs;
  real Mobs[nobs];
  real sigma_M[nobs];
  real ageobs[nobs];
  real sigma_age[nobs];
}

transformed data {
  real x_r[2 + nltc];
  int x_i[1];
  real Mintegrate_max[1];

  Mintegrate_max[1] = 1000.0; /* This is probably the largest mass we
				 can reasonably deal with, so let's
				 just integrate up to here. */

  x_i[1] = nltc;

  x_r[1] = MMin;
  x_r[2] = tmax;
  for (i in 1:nltc) {
    x_r[2+i] = ltc[i];
  }
}

parameters {
  real<lower=0> L; /* Count / number of stars formed (detected + undetected) */

  real<lower=1> alpha; /* Power-law index on IMF: dN/dM ~ M^-alpha */
  
  real<lower=0,upper=tmax> mu_t; /* Mean of Gaussian-shape for SFR */
  real<lower=0,upper=tmax> sigma_t; /* S.D. of Gaussian-shape SFR. */
  
  real<lower=MMin> Mtrue[nobs];
  real<lower=0, upper=1> fttrue[nobs]; /* True age is age_min +
					  ftrue*(age_max-age_min) */
}

transformed parameters {
  real agetrue[nobs];
  real agelogJfactors[nobs];

  for (i in 1:nobs) {
    /* Jacobian for age.  Because the sampling variable is f =
       age/lifetime, we need to multiply if d(age)/df = 1/(df/d(age))
       = lifetime  (or dt for the smaller masses). */
    real lt = lifetime(Mtrue[i], ltc);
    agetrue[i] = fttrue[i]*lt;

    agelogJfactors[i] = log(lt);
  }
}

model {
  real fobs; /* Fraction of systems that are observable. */

  real theta[3];
  real state0[1];
  real state_result[1,1];

  theta[1] = alpha;
  theta[2] = mu_t;
  theta[3] = sigma_t;

  state0[1] = 0.0;

  state_result = integrate_ode_rk45(dfdM, state0, MMin, Mintegrate_max, theta, x_r, x_i);
  fobs = state_result[1,1];

  /* Priors */
  L ~ cauchy(0.0, nobs);

  alpha ~ cauchy(0.0, 1.0);

  /* mu_t flat between limits */
  sigma_t ~ cauchy(0.0, tmax/2.0);
  
  /* Observed systems */
  /* Likelihood terms */
  Mobs ~ normal(Mtrue, sigma_M);
  ageobs ~ normal(agetrue, sigma_age);
  /* Hierarchical (Population) Prior */
  target += nobs*log(L);
  target += sum(log_dPdMdt(Mtrue, agetrue, alpha, mu_t, sigma_t, MMin, tmax));
  target += sum(agelogJfactors);

  target += -L*fobs; /* Poisson normalisation, by expected *detected* systems. */
}
