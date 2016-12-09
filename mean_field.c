#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>

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

int read_par_mean_field(int argc, char ** argv,
			size_t * n_neurons,
			double * p_e, double * w_e_min,
			double * w_e_max,
			double * tau_e, size_t * d_e,
			double * p_i, double * w_i_min,
			double * w_i_max,
			double * tau_i, size_t * d_i,
			double * varphi_0, double * varphi_k,
			size_t * n_steps);

void print_usage_mean_field();

int write_mean_field_preamble(size_t * n_neurons,
			      double * p_e, double * w_e_min,
			      double * w_e_max,
			      double * tau_e, size_t * d_e,
			      double * p_i, double * w_i_min,
			      double * w_i_max,
			      double * tau_i, size_t * d_i,
			      double * varphi_0, double * varphi_k,
			      size_t * n_steps);

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

int main(int argc, char ** argv)
{
  size_t n_neurons, d_e, d_i, n_steps;
  double p_e, w_e_min, w_e_max, tau_e;
  double p_i, w_i_min, w_i_max, tau_i;
  double varphi_0, varphi_k;
  int status = read_par_mean_field(argc, argv, &n_neurons,
				   &p_e, &w_e_min, &w_e_max,
				   &tau_e, &d_e,
				   &p_i, &w_i_min, &w_i_max,
				   &tau_i, &d_i,
				   &varphi_0, &varphi_k, &n_steps);

  if (status == -1) exit (EXIT_FAILURE);

  write_mean_field_preamble(&n_neurons, &p_e, &w_e_min, &w_e_max,
			    &tau_e, &d_e, &p_i, &w_i_min, &w_i_max,
			    &tau_i, &d_i, &varphi_0, &varphi_k,
			    &n_steps);
  gsl_vector * Ig_e;
  Ig_e = G_e(tau_e, d_e);
  gsl_vector * Ig_i;
  Ig_i = G_i(tau_i, d_i);
  double step = (1-varphi_0)/(n_steps-1);
  for (size_t i=0; i<n_steps; i++)
    {
      double t = varphi_0+i*step;
      double u = u_at_nu(t, n_neurons,
			 p_e, w_e_min, w_e_max, Ig_e,
			 p_i, w_i_min, w_i_max, Ig_i);
      double varphi_val = varphi(u,varphi_0,varphi_k);
      fprintf(stdout,"%12.10g\t%12.10g\t%12.10g\n",
	      t,u,varphi_val);
    }

  gsl_vector_free(Ig_e);
  gsl_vector_free(Ig_i);
  exit (EXIT_SUCCESS);
}

int read_par_mean_field(int argc, char ** argv,
			size_t * n_neurons,
			double * p_e, double * w_e_min, double * w_e_max,
			double * tau_e, size_t * d_e,
			double * p_i, double * w_i_min, double * w_i_max,
			double * tau_i, size_t * d_i,
			double * varphi_0, double * varphi_k,
			size_t * n_steps)
{
  if (argc == 1) {
    print_usage_mean_field();
    return -1;
  }
  // Define default values
  *p_e = 0.1;
  *w_e_min = 2;
  *w_e_max = 4;
  *tau_e = 5;
  *d_e = 1;
  *p_i = 0.2;
  *w_i_min = -0.01;
  *w_i_max = -0.005;
  *tau_i = 5;
  *d_i = 5;
  *varphi_0=0.01;
  *varphi_k = 10;
  *n_steps = 1001;
  {int opt;
    static struct option long_options[] = {
      {"n_neurons",required_argument,NULL,'n'},
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
      {"n_steps",optional_argument,NULL,'o'},
      {NULL,0,NULL,0}
    };
    int long_index =0;
    while ((opt = getopt_long(argc,argv,"hn:a:b:c:d:e:f:g:i:j:k:l:m:o:",
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
	  *n_neurons=(size_t) n; 
	}
	break;
      case 'a': *p_e=atof(optarg);
	break;
      case 'b': *w_e_min=atof(optarg);
	break;
      case 'c': *w_e_max=atof(optarg);
	break;
      case 'd': *p_i=atof(optarg);
	break;
      case 'e': *w_i_min=atof(optarg);
	break;
      case 'f': *w_i_max=atof(optarg);
	break;
      case 'i': *tau_e=atof(optarg);
	break;
      case 'j':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The excitatory synaptic delay should be > 0.\n");
	      return -1;
	    }
	  *d_e=(size_t) n; 
	}
	break;
      case 'k': *tau_i=atof(optarg);
	break;
      case 'l':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The inhibitory synaptic delay should be > 0.\n");
	      return -1;
	    }
	  *d_i=(size_t) n; 
	}
	break;
      case 'm': *varphi_0=atof(optarg);
	break;
      case 'g': *varphi_k=atof(optarg);
	break;
      case 'o':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The number of nu_bar steps should be > 0.\n");
	      return -1;
	    }
	  *n_steps=(size_t) n; 
	}
	break;
      case 'h': print_usage_mean_field();
	return -1;
      default : print_usage_mean_field();
	return -1;
      }
    }
  }
  // Check that the parameters are correct
  if (*p_e < 0 || *p_e > 1)
    {
      fprintf(stderr,"We must have 0 <= p_e <= 1.\n");
      return -1;
    }
  if (*p_i < 0 || *p_i > 1)
    {
      fprintf(stderr,"We must have 0 <= p_i <= 1.\n");
      return -1;
    }
  if (*w_e_min <= 0 || *w_e_max <= 0)
    {
      fprintf(stderr,"Excitatory weights must be > 0.\n");
      return -1;
    }
  if (*w_e_min >= *w_e_max)
    {
      fprintf(stderr,"We must have w_e_min < w_e_max.\n");
      return -1;
    }
  if (*w_i_min >= 0 || *w_i_max >= 0)
    {
      fprintf(stderr,"Inhibitory weights must be < 0.\n");
      return -1;
    }
  if (*w_i_min >= *w_i_max)
    {
      fprintf(stderr,"We must have w_i_min < w_i_max.\n");
      return -1;
    }
  if (*tau_i < 0)
    {
      fprintf(stderr,"We must have 0 <= tau_i.\n");
      return -1;
    }
  if (*tau_e < 0)
    {
      fprintf(stderr,"We must have 0 <= tau_e.\n");
      return -1;
    }
  if (*varphi_0 < 0 || *varphi_0 > 1)
    {
      fprintf(stderr,"We must have 0 <= varphi_0 <= 1.\n");
      return -1;
    }
  if (*varphi_k <= 0)
    {
      fprintf(stderr,"We must have 0 < varphi_k.\n");
      return -1;
    }
  return 0;
}

void print_usage_mean_field() {
  printf("Usage: \n"
	 "  --n_neurons <positive integer>: the number of neurons in the"
	 " network\n"
	 "  --p_e <double in (0,1)>: the probability of excitatory "
	 "connection between two neurons\n"
	 "  --w_e_min <positive double>: the minimal excitatory synaptic "
	 "weight\n"
	 "  --w_e_min <positive double>: the maximal excitatory synaptic "
	 "weight\n"
	 "  --tau_e <positive double>: the time constant of excitatory "
	 "leak functions\n"
	 "  --d_e <positive integer>: the excitatory synaptic delay\n"
	 "  --p_i <double in (0,1)>: the probability of inhibitory "
	 "connection between two neurons\n"
	 "  --w_i_min <negative double>: the minimal inhibitory synaptic "
	 "weight\n"
	 "  --w_i_max <negative double>: the maximal inhibitory synaptic "
	 "weight\n"
	 "  --tau_i <positive double>: the time constant of inhibitory "
	 "leak functions\n"
	 "  --d_i <positive integer>: the inhibitory synaptic delay\n"
	 "  --varphi_0 <double in (0,1)>: the basal value of the "
	 "activation function\n"
	 "  --varphi_k <positive double>: constant controlling the "
	 "steepness of the activation function\n"
	 "  --n_steps <positive integer>: the number of nu_bar values "
	 "to explore between varphi_0 and 1\n"
	 "\n");
}

int write_mean_field_preamble(size_t * n_neurons,
			      double * p_e, double * w_e_min,
			      double * w_e_max,
			      double * tau_e, size_t * d_e,
			      double * p_i, double * w_i_min,
			      double * w_i_max,
			      double * tau_i, size_t * d_i,
			      double * varphi_0, double * varphi_k,
			      size_t * n_steps)
{
  fprintf(stdout,"###########################################\n"
	  "# Parameters used when running the program\n");
  fprintf(stdout,"# The number of neurons is: %d\n", (int) * n_neurons);
  fprintf(stdout,"# Probability of excitatory connection: %g\n", * p_e);
  fprintf(stdout,"# Minimal excitatory weight: %g\n", * w_e_min);
  fprintf(stdout,"# Maximal excitatory weight: %g\n", * w_e_max);
  fprintf(stdout,"# Excitatory time constant: %g\n", * tau_e);
  fprintf(stdout,"# Excitatory time delay: %d\n", (int) * d_e);
  fprintf(stdout,"# Probability of inhibitory connection: %g\n", * p_i);
  fprintf(stdout,"# Minimal inhibitory weight: %g\n", * w_i_min);
  fprintf(stdout,"# Maximal inhibitory weight: %g\n", * w_i_max);
  fprintf(stdout,"# Inhibitory time constant: %g\n", * tau_i);
  fprintf(stdout,"# Inhibitory time delay: %d\n", (int) * d_i);
  fprintf(stdout,"# varphi_0: %g\n", * varphi_0);
  fprintf(stdout,"# varphi_k: %g\n", * varphi_k);
  fprintf(stdout,"# Number of steps: %d\n", (int) * n_steps);
  fprintf(stdout,"###########################################\n");
  return 0;
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
