#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_roots.h>

#include <math.h>

#if defined(FAST_EXP)
static union
{
  double d;
  struct
  {
    int j,i;
  } n;
} _eco;

#define EXP_A (1048576/M_LN2)
#define EXP_C 60801

#define EXP(y) (_eco.n.i=EXP_A*(y)+(1072693248-EXP_C),_eco.d)
#endif

typedef struct
{
  size_t n_neurons, d_e, d_i;
  double p_e, w_e_min, w_e_max, p_i, w_i_min, w_i_max, varphi_0, \
    varphi_k, tau_e, tau_i;
  gsl_vector * G_e;
  gsl_vector * G_i;
} mean_field_fixed_point_params;

double mf_fixed_point_target(double nu, void *params);


int read_par_mean_field_fixed_point(int argc, char ** argv,
				    mean_field_fixed_point_params *p,
				    double * nu_lower,
				    double * nu_upper);

void print_usage_mean_field_fixed_point();


double u_at_nu(double nu_bar, size_t n_neurons,
	       double p_e, double w_e_min, double w_e_max,
	       gsl_vector * G_e,
	       double p_i, double w_i_min, double w_i_max,
	       gsl_vector * G_i);

gsl_vector * G_i(double tau_i, size_t d_i);

gsl_vector * G_e(double tau_e, size_t d_e);

double varphi(double u, double varphi_0, double k);

double g_i(size_t delay, double tau_i, size_t d_i);

double g_e(size_t delay, double tau_e, size_t d_e);

double mf_fixed_point_target(double nu, void *params);

int main(int argc, char ** argv)
{
  mean_field_fixed_point_params params;
  double nu_lower, nu_upper;
  int status = read_par_mean_field_fixed_point(argc, argv, &params,
					       &nu_lower, &nu_upper);

  if (status == -1) exit (EXIT_FAILURE);
  
  params.G_e = G_e(params.tau_e, params.d_e);
  params.G_i = G_i(params.tau_i, params.d_i);

  int iter = 0, max_iter = 100;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  gsl_function F;
  F.function = &mf_fixed_point_target;
  F.params = &params;

  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc (T);
  gsl_root_fsolver_set (s, &F, nu_lower, nu_upper);
  printf ("using %s method\n",
	  gsl_root_fsolver_name (s));

  printf ("%5s [%9s, %9s] %9s %9s\n",
	  "iter", "lower", "upper", "root",
	  "err(est)");

  do
    {
      iter++;
      status = gsl_root_fsolver_iterate (s);
      double r = gsl_root_fsolver_root (s);
      nu_lower = gsl_root_fsolver_x_lower (s);
      nu_upper = gsl_root_fsolver_x_upper (s);
      status = gsl_root_test_interval (nu_lower, nu_upper,
				       0, 0.001);

      if (status == GSL_SUCCESS)
	printf ("Converged:\n");

      printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
	      iter, nu_lower, nu_upper,
	      r, nu_upper - nu_lower);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_root_fsolver_free (s);
  gsl_vector_free(params.G_e);
  gsl_vector_free(params.G_i);
  return status;
}

int read_par_mean_field_fixed_point(int argc, char ** argv,
				    mean_field_fixed_point_params *p,
				    double * nu_lower,
				    double * nu_upper)
{
  // Define default values
  
  p->n_neurons = 800;
  p->p_e = 0.1;
  p->w_e_min = 0.5;
  p->w_e_max = 1;
  p->tau_e = 5;
  p->d_e = 1;
  p->p_i = 0.2;
  p->w_i_min = -0.004;
  p->w_i_max = -0.002;
  p->tau_i = 5;
  p->d_i = 4;
  p->varphi_0 = 0.01;
  p->varphi_k = 15;
  *nu_lower = 0.25;
  *nu_upper = 0.3;
  {int opt;
    static struct option long_options[] = {
      {"n_neurons",optional_argument,NULL,'n'},
      {"p_e",optional_argument,NULL,'a'},
      {"w_e_min",optional_argument,NULL,'b'},
      {"w_e_max",optional_argument,NULL,'c'},
      {"p_i",optional_argument,NULL,'d'},
      {"w_i_min",optional_argument,NULL,'e'},
      {"w_i_max",optional_argument,NULL,'f'},
      {"help",no_argument,NULL,'h'},
      {"tau_e",optional_argument,NULL,'i'},
      {"d_e",optional_argument,NULL,'j'},
      {"tau_i",optional_argument,NULL,'k'},
      {"d_i",optional_argument,NULL,'l'},
      {"varphi_0",optional_argument,NULL,'m'},
      {"varphi_k",optional_argument,NULL,'g'},
      {"nu_lower",optional_argument,NULL,'o'},
      {"nu_upper",optional_argument,NULL,'p'},
      {NULL,0,NULL,0}
    };
    int long_index =0;
    while ((opt = getopt_long(argc,argv,
			      "hn:a:b:c:d:e:f:g:i:j:k:l:m:o:p:",
			      long_options,&long_index)) != -1) {
      switch(opt) {
      case 'n':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The number of neurons should be > 0.\n");
	      return -1;
	    }
	  p->n_neurons=(size_t) n; 
	}
	break;
      case 'a': p->p_e=atof(optarg);
	break;
      case 'b': p->w_e_min=atof(optarg);
	break;
      case 'c': p->w_e_max=atof(optarg);
	break;
      case 'd': p->p_i=atof(optarg);
	break;
      case 'e': p->w_i_min=atof(optarg);
	break;
      case 'f': p->w_i_max=atof(optarg);
	break;
      case 'i': p->tau_e=atof(optarg);
	break;
      case 'j':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The excitatory synaptic delay should "
		      "be > 0.\n");
	      return -1;
	    }
	  p->d_e=(size_t) n; 
	}
	break;
      case 'k': p->tau_i=atof(optarg);
	break;
      case 'l':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The inhibitory synaptic delay should "
		      "be > 0.\n");
	      return -1;
	    }
	  p->d_i=(size_t) n; 
	}
	break;
      case 'm': p->varphi_0=atof(optarg);
	break;
      case 'g': p->varphi_k=atof(optarg);
	break;
      case 'o': *nu_lower=atof(optarg);
	break;
      case 'p': *nu_upper=atof(optarg);
	break;
      case 'h': print_usage_mean_field_fixed_point();
	return -1;
      default : print_usage_mean_field_fixed_point();
	return -1;
      }
    }
  }
  // Check that the parameters are correct
  if (p->p_e < 0 || p->p_e > 1)
    {
      fprintf(stderr,"We must have 0 <= p_e <= 1.\n");
      return -1;
    }
  if (p->p_i < 0 || p->p_i > 1)
    {
      fprintf(stderr,"We must have 0 <= p_i <= 1.\n");
      return -1;
    }
  if (p->w_e_min <= 0 || p->w_e_max <= 0)
    {
      fprintf(stderr,"Excitatory weights must be > 0.\n");
      return -1;
    }
  if (p->w_e_min >= p->w_e_max)
    {
      fprintf(stderr,"We must have w_e_min < w_e_max.\n");
      return -1;
    }
  if (p->w_i_min >= 0 || p->w_i_max >= 0)
    {
      fprintf(stderr,"Inhibitory weights must be < 0.\n");
      return -1;
    }
  if (p->w_i_min >= p->w_i_max)
    {
      fprintf(stderr,"We must have w_i_min < w_i_max.\n");
      return -1;
    }
  if (p->tau_i < 0)
    {
      fprintf(stderr,"We must have 0 <= tau_i.\n");
      return -1;
    }
  if (p->tau_e < 0)
    {
      fprintf(stderr,"We must have 0 <= tau_e.\n");
      return -1;
    }
  if (p->varphi_0 < 0 || p->varphi_0 > 1)
    {
      fprintf(stderr,"We must have 0 <= varphi_0 <= 1.\n");
      return -1;
    }
  if (p->varphi_k <= 0)
    {
      fprintf(stderr,"We must have 0 < varphi_k.\n");
      return -1;
    }
  if (*nu_lower < 0 || *nu_lower > 1)
    {
      fprintf(stderr,"We must have 0 <= nu_lower <= 1.\n");
      return -1;
    }
  if (*nu_upper < 0 || *nu_upper > 1 || *nu_upper <= *nu_lower)
    {
      fprintf(stderr,"We must have 0 <= nu_lower < nu_lower <= 1.\n");
      return -1;
    }
  return 0;
}

void print_usage_mean_field_fixed_point() {
  printf("Usage: \n"
	 "  --n_neurons <positive integer>: the number of neurons in "
	 "the network\n"
	 "  --p_e <double in (0,1)>: the probability of excitatory "
	 "connection between two neurons\n"
	 "  --w_e_min <positive double>: the minimal excitatory "
	 "synaptic weight\n"
	 "  --w_e_min <positive double>: the maximal excitatory "
	 "synaptic weight\n"
	 "  --tau_e <positive double>: the time constant of "
	 "excitatory leak functions\n"
	 "  --d_e <positive integer>: the excitatory synaptic delay\n"
	 "  --p_i <double in (0,1)>: the probability of inhibitory "
	 "connection between two neurons\n"
	 "  --w_i_min <negative double>: the minimal inhibitory "
	 "synaptic weight\n"
	 "  --w_i_max <negative double>: the maximal inhibitory "
	 "synaptic weight\n"
	 "  --tau_i <positive double>: the time constant of "
	 "inhibitory leak functions\n"
	 "  --d_i <positive integer>: the inhibitory synaptic delay\n"
	 "  --varphi_0 <double in (0,1)>: the basal value of the "
	 "activation function\n"
	 "  --varphi_k <positive double>: constant controlling the "
	 "steepness of the activation function\n"
	 "  --nu_lower <double in (0,1)>: the left end of the root "
	 "bracketing interval\n"
	 "  --nu_upper <double in (0,1)>: the right end of the root "
	 "bracketing interval\n"
	 "\n");
}

double u_at_nu(double nu_bar, size_t n_neurons,
	       double p_e, double w_e_min, double w_e_max,
	       gsl_vector * G_e,
	       double p_i, double w_i_min, double w_i_max,
	       gsl_vector * G_i)
{
  size_t max = G_i->size;
  double u = 0;
  double e_factor = p_e*(w_e_min+w_e_max)*0.5;
  double i_factor = p_i*(w_i_min+w_i_max)*0.5;
  for (size_t s=2; s <= max; s++)
    {
      double Ge;
      if (s-1 < G_e->size)
	Ge = gsl_vector_get(G_e,s-1);
      else
	Ge = gsl_vector_get(G_e,G_e->size-1);
      double Gi = gsl_vector_get(G_i,s-1);
      u += pow((1-nu_bar),(double) (s-1))*(Ge*e_factor+Gi*i_factor);
    }
  return u*n_neurons*nu_bar*nu_bar;
}

gsl_vector * G_i(double tau_i, size_t d_i)
{
  size_t n = ceil(10*tau_i);
  gsl_vector * res = gsl_vector_alloc(n);
  gsl_vector_set(res,0,g_i(1,tau_i,d_i));
  for (size_t i=1; i < n; i++)
    gsl_vector_set(res,i,gsl_vector_get(res,i-1)+g_i(i+1,tau_i,d_i));
  return res;
}

gsl_vector * G_e(double tau_e, size_t d_e)
{
  size_t n = ceil(5*tau_e);
  gsl_vector * res = gsl_vector_alloc(n);
  gsl_vector_set(res,0,g_e(1,tau_e,d_e));
  for (size_t i=1; i < n; i++)
    gsl_vector_set(res,i,gsl_vector_get(res,i-1)+g_e(i+1,tau_e,d_e));
  return res;
}

double g_i(size_t delay, double tau_i, size_t d_i)
{
  double x = (delay-d_i)/tau_i;
  if (x < 0 || x > 10)
    return 0.0;
  else
    {
#if defined(FAST_EXP)
      return x*EXP(1-x);
#else
      return x*exp(1-x);
#endif
    }
}

double g_e(size_t delay, double tau_e, size_t d_e)
{
  double x = (delay-d_e)/tau_e;
  if (x < 0 || x > 5)
    return 0.0;
  else
    {
#if defined(FAST_EXP)
      return EXP(-x);
#else
      return exp(-x);
#endif      
    }
}

double varphi(double u, double varphi_0, double k)
{
  if (u < 0)
    return varphi_0;
  else
    {
#if defined(FAST_EXP)
      return varphi_0 + (1-varphi_0)*gsl_pow_2 (1-EXP(-u/k));
#else
      return varphi_0 + (1-varphi_0)*gsl_pow_2 (1-exp(-u/k));
#endif
    }
}

double mf_fixed_point_target(double nu, void *params)
{
  mean_field_fixed_point_params *p = \
    (mean_field_fixed_point_params *) params;
  size_t n_neurons = p->n_neurons;
  double p_e = p->p_e;
  double w_e_min = p->w_e_min;
  double w_e_max = p->w_e_max;
  double p_i = p->p_i;
  double w_i_min = p->w_i_min;
  double w_i_max = p->w_i_max;
  double varphi_0 = p->varphi_0;
  double varphi_k = p->varphi_k;
  gsl_vector * G_e = p->G_e;
  gsl_vector * G_i = p->G_i;

  double u = u_at_nu(nu, n_neurons, p_e, w_e_min, w_e_max, G_e,
		     p_i, w_i_min, w_i_max, G_i);
  return varphi(u,varphi_0,varphi_k)-nu;
}
