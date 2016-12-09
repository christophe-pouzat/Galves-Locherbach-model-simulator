#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_uint.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <glib.h>

typedef struct
{
  size_t size;
  gsl_vector_uint * idx;
  gsl_vector * w; 
} presynaptic;

presynaptic ** mk_graph(size_t n_neurons,
			double p_e, double w_e_min, double w_e_max,
			double p_i, double w_i_min, double w_i_max,
			gsl_rng * r);

int free_graph(presynaptic **graph, size_t n_neurons);

GArray ** malloc_garrays2(size_t n_neurons);

int free_garrays2(GArray ** history, size_t n_neurons);

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
  size_t n_neurons, d_e, d_i, total_steps, early_steps;
  double p_e, w_e_min, w_e_max, p_i, w_i_min, w_i_max, varphi_0, \
    varphi_k, tau_e, tau_i, nu_bar;
  gsl_vector * G_e;
  gsl_vector * G_i;
} sim_gl_params;

int write_sim_gl_results(size_t n_neurons, GArray **history);

int write_sim_gl_preamble(sim_gl_params * params, gsl_rng * r);

int mk_one_step(int t,
		GArray **history,
		presynaptic **graph,
		sim_gl_params * params,
		gsl_rng * r);

int read_par_sim_gl(int argc, char ** argv, sim_gl_params *p);

void print_usage_sim_gl();

int spike_or_not(size_t n_idx, int t,
		 GArray **history, presynaptic **graph,
		 double tau_e, size_t d_e,
		 double tau_i, size_t d_i,
		 double varphi_0, double k,
		 gsl_rng * r);

double varphi(double u, double varphi_0, double k);

double get_u_i(size_t n_idx, int t,
	       GArray **history, presynaptic **graph,
	       double tau_e, size_t d_e,
	       double tau_i, size_t d_i);

int spike_or_not_early(double nu_bar,
		       gsl_rng * r);

double g_i(size_t delay, double tau_i, size_t d_i);

double g_e(size_t delay, double tau_e, size_t d_e);

gsl_vector * G_i(double tau_i, size_t d_i);

gsl_vector * G_e(double tau_e, size_t d_e);

int main(int argc, char ** argv)
{
  sim_gl_params params;
  // Read and check parameters
  int status = read_par_sim_gl(argc, argv, &params);

  if (status == -1) exit (EXIT_FAILURE);
  
  params.G_e = G_e(params.tau_e, params.d_e);
  params.G_i = G_i(params.tau_i, params.d_i);

  // Initialize RNG
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  // Generate network
  presynaptic **graph=mk_graph(params.n_neurons,
			       params.p_e, params.w_e_min,
			       params.w_e_max,
			       params.p_i, params.w_i_min,
			       params.w_i_max, r);

  // Allocate history
  GArray **history = malloc_garrays2 (params.n_neurons);

  // Write preamble
  write_sim_gl_preamble(&params, r);
  
  // Do the job
  for (int step_idx=0; step_idx < (int) params.total_steps; step_idx++)
    {
      mk_one_step(step_idx, history, graph, &params, r);
    }

  // Write results
  write_sim_gl_results(params.n_neurons, history);
  
  // Free memory taken up by history
  free_garrays2(history,params.n_neurons);
  
  gsl_rng_free (r);
  gsl_vector_free(params.G_e);
  gsl_vector_free(params.G_i);
  free_graph(graph, params.n_neurons);
  exit (EXIT_SUCCESS);
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

int spike_or_not_early(double nu_bar,
		       gsl_rng * r)
{
  if (gsl_ran_flat(r,0.0,1.0) <= nu_bar)
    return 1;
  else
    return 0;
}

double get_u_i(size_t n_idx, int t,
	       GArray **history, presynaptic **graph,
	       double tau_e, size_t d_e,
	       double tau_i, size_t d_i)
{
  double u_i=0.0;
  // Get the time of the last spike of n_idx
  int L_i = g_array_index(history[n_idx],int,history[n_idx]->len-1);
  if (graph[n_idx]->size > 0)
    {// n_idx has presynaptic neurons
      for (size_t pre_idx=0; pre_idx<graph[n_idx]->size; pre_idx++)
	{// Loop on the presynaptic neurons
	  // Get the index of the presynaptic neuron
	  uint j = gsl_vector_uint_get(graph[n_idx]->idx,pre_idx);
	  // Get the synaptic weight
	  double w = gsl_vector_get(graph[n_idx]->w,pre_idx);
	  // Get the index of the last spike of neuron j
	  size_t k = history[j]->len-1;
	  // Get the time of the last spike of j
	  int s = g_array_index(history[j],int,k);
	  while (s > L_i)
	    {
	      if (w > 0) //excitatory synapse
		u_i += w*g_e(t-s,tau_e,d_e);
	      else //inhibitory synapse
		u_i += w*g_i(t-s,tau_i,d_i);
	      k--;
	      if (k<0)
		s = L_i;
	      else
		s = g_array_index(history[j],int,k);
	    } // end of conditional on s > L_i  
	} // end of the loop on pre_idx
    } // end of the conditional on graph[n_idx]->size > 0
  return u_i;
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

int spike_or_not(size_t n_idx, int t,
		 GArray **history, presynaptic **graph,
		 double tau_e, size_t d_e,
		 double tau_i, size_t d_i,
		 double varphi_0, double k,
		 gsl_rng * r)
{
  double u = get_u_i(n_idx, t, history, graph,
		     tau_e, d_e, tau_i, d_i);
  if (gsl_ran_flat(r,0.0,1.0) <= varphi(u, varphi_0, k))
    return 1;
  else
    return 0;
}

void print_usage_sim_gl() {
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
	 "  --nu_bar <double in (0,1)>: the mean field rate\n"
	 "  --early_steps <positive integer>: the number of time "
	 "steps with IID draws\n"
	 "  --total_steps <positive integer>: the total number of "
	 "time steps to simulate\n"
	 "\n");
}

int read_par_sim_gl(int argc, char ** argv, sim_gl_params *p)
{
  // Define default values
  
  p->n_neurons = 800;
  p->p_e = 0.1;
  p->w_e_min = 0.2;
  p->w_e_max = 0.3;
  p->tau_e = 5;
  p->d_e = 1;
  p->p_i = 0.25;
  p->w_i_min = -0.02;
  p->w_i_max = -0.005;
  p->tau_i = 5;
  p->d_i = 4;
  p->varphi_0 = 0.01;
  p->varphi_k = 17;
  p->nu_bar = 0.2217;
  p->early_steps = 100;
  p->total_steps = 60000;
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
      {"nu_bar",optional_argument,NULL,'o'},
      {"early_steps",optional_argument,NULL,'p'},
      {"total_steps",optional_argument,NULL,'q'},
      {NULL,0,NULL,0}
    };
    int long_index =0;
    while ((opt = getopt_long(argc,argv,
			      "hn:a:b:c:d:e:f:g:i:j:k:l:m:o:p:q:",
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
      case 'o': p->nu_bar=atof(optarg);
	break;
      case 'p':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The number of early steps should be "
		      "> 0.\n");
	      return -1;
	    }
	  p->early_steps=(size_t) n; 
	}
	break;
      case 'q':
	{
	  int n=atoi(optarg);
	  if (n <= 0)
	    {
	      fprintf(stderr,"The total number of steps should be "
		      "> 0.\n");
	      return -1;
	    }
	  p->total_steps=(size_t) n; 
	}
	break;
      case 'h': print_usage_sim_gl();
	return -1;
      default : print_usage_sim_gl();
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
  if (p->nu_bar < 0 || p->nu_bar > 1)
    {
      fprintf(stderr,"We must have 0 <= nu_bar <= 1.\n");
      return -1;
    }
  return 0;
}

int mk_one_step(int t,
		GArray **history,
		presynaptic **graph,
		sim_gl_params * params,
		gsl_rng * r)
{
  int res[params->n_neurons];
  // Spike or no spike for each neuron
  if (t < params->early_steps)
    {
      for (size_t n_idx=0; n_idx<params->n_neurons; n_idx++)
	res[n_idx] = spike_or_not_early(params->nu_bar,r);
    }
  else
    {
      for (size_t n_idx=0; n_idx<params->n_neurons; n_idx++)
	res[n_idx] = spike_or_not(n_idx, t, history, graph,
				  params->tau_e,params->d_e,
				  params->tau_i,params->d_i,
				  params->varphi_0,params->varphi_k,
				  r);
      
    }
  // Upate history
  for (size_t n_idx = 0; n_idx < params->n_neurons; n_idx++)
    {
      if (res[n_idx] == 1)
	g_array_append_val(history[n_idx],t);
    }
  return 0;
}

int write_sim_gl_preamble(sim_gl_params * params, gsl_rng * r)
{
  fprintf(stdout,"###########################################\n"
	  "# Parameters used when running the program\n");
  fprintf(stdout,"# The number of neurons is: %d\n",
	  (int) params->n_neurons);
  fprintf(stdout,"# Probability of excitatory connection: %g\n",
	  params->p_e);
  fprintf(stdout,"# Minimal excitatory weight: %g\n", params->w_e_min);
  fprintf(stdout,"# Maximal excitatory weight: %g\n", params->w_e_max);
  fprintf(stdout,"# Excitatory time constant: %g\n", params->tau_e);
  fprintf(stdout,"# Excitatory time delay: %d\n", (int) params->d_e);
  fprintf(stdout,"# Probability of inhibitory connection: %g\n",
	  params->p_i);
  fprintf(stdout,"# Minimal inhibitory weight: %g\n", params->w_i_min);
  fprintf(stdout,"# Maximal inhibitory weight: %g\n", params->w_i_max);
  fprintf(stdout,"# Inhibitory time constant: %g\n", params->tau_i);
  fprintf(stdout,"# Inhibitory time delay: %d\n", (int) params->d_i);
  fprintf(stdout,"# varphi_0: %g\n", params->varphi_0);
  fprintf(stdout,"# varphi_k: %g\n", params->varphi_k);
  fprintf(stdout,"# nu_bar: %g\n", params->nu_bar);
  fprintf(stdout,"# early_steps: %d\n", (int) params->early_steps);
  fprintf(stdout,"# total_steps: %d\n", (int) params->total_steps);
  fprintf(stdout,"#\n");
  fprintf(stdout,"# Generator type: %s\n", gsl_rng_name (r));
  fprintf(stdout,"# Seed = %lu\n", gsl_rng_default_seed);
  fprintf(stdout,"###########################################\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"\n");
  return 0;
}

int write_sim_gl_results(size_t n_neurons, GArray **history)
{
  for (size_t n_idx = 0; n_idx < n_neurons; n_idx++)
    {
      fprintf(stdout,"# Start neuron %d with %d spikes\n",
	      (int) n_idx, history[n_idx]->len);
      for (size_t s_idx = 0; s_idx < history[n_idx]->len; s_idx++)
	{
	  fprintf(stdout,"%d\n", g_array_index(history[n_idx],int,s_idx));
	}
      fprintf(stdout,"# End neuron %d\n",(int) n_idx);
      fprintf(stdout,"\n");
      fprintf(stdout,"\n");
    }
  return 0;
}

GArray ** malloc_garrays2(size_t n_neurons)
{
  GArray **result=malloc(n_neurons*sizeof(GArray*));
  for (size_t n_idx=0; n_idx < n_neurons; n_idx++)
    {
      result[n_idx] = g_array_sized_new(FALSE, FALSE, sizeof(int), 1024);
    }
  return result;  
}

int free_garrays2(GArray ** history, size_t n_neurons)
{
  for (size_t n_idx=0; n_idx < n_neurons; n_idx++)
    {
      g_array_free(history[n_idx],TRUE);
    }
  free(history);
  return 0;
}

presynaptic ** mk_graph(size_t n_neurons,
			double p_e, double w_e_min, double w_e_max,
			double p_i, double w_i_min, double w_i_max,
			gsl_rng * r)
{
  // Check that the parameters are correct
  if (p_e < 0 || p_e > 1)
    {
      fprintf(stderr,"We must have 0 <= p_e <= 1.\n");
      return NULL;
    }
  if (p_i < 0 || p_i > 1)
    {
      fprintf(stderr,"We must have 0 <= p_i <= 1.\n");
      return NULL;
    }
  if (w_e_min <= 0 || w_e_max <= 0)
    {
      fprintf(stderr,"Excitatory weights must be > 0.\n");
      return NULL;
    }
  if (w_e_min >= w_e_max)
    {
      fprintf(stderr,"We must have w_e_min < w_e_max.\n");
      return NULL;
    }
  if (w_i_min >= 0 || w_i_max >= 0)
    {
      fprintf(stderr,"Inhibitory weights must be < 0.\n");
      return NULL;
    }
  if (w_i_min >= w_i_max)
    {
      fprintf(stderr,"We must have w_i_min < w_i_max.\n");
      return NULL;
    }
  // allocate memory for the result
  presynaptic **graph = malloc(n_neurons*sizeof(presynaptic*));
  for (size_t n_idx=0; n_idx < n_neurons; n_idx++)
    { // For each postsynaptic neuron
      uint idx[n_neurons*2];
      double w[n_neurons*2];
      size_t n=0; // counts the number of actual connections
      for (size_t pre_idx=0; pre_idx < n_neurons; pre_idx++)
	{ // For each potential presynaptic neuron
	  if (pre_idx == n_idx) continue; // No autapses!
	  if(gsl_ran_flat(r,0.0,1.0) <= p_e)
	    {// there is an excitatory connection
	      idx[n]=pre_idx; // add it to the list
	      w[n]=gsl_ran_flat(r,w_e_min,w_e_max); // draw its weight
	      n++; // increase n by 1
	    }
	  if(gsl_ran_flat(r,0.0,1.0) <= p_i)
	    {// there is an inhibitory connection
	      idx[n]=pre_idx; // add it to the list
	      w[n]=gsl_ran_flat(r,w_i_min,w_i_max); // draw its weight
	      n++; // increase n by 1
	    }
	} // End of loop on each potential presynaptic neuron
      // Initialize the presynaptic structure for neuron n_idx
      // Start by allocating the required memory
      graph[n_idx] = malloc(sizeof(presynaptic));
      graph[n_idx]->size=n;
      if (n > 0)
	{
	 graph[n_idx]->w = gsl_vector_alloc(n);
	 graph[n_idx]->idx = gsl_vector_uint_alloc(n);
	 // Assign values
	 for (size_t pre_idx=0; pre_idx < n; pre_idx++)
	   {
	     gsl_vector_uint_set(graph[n_idx]->idx,pre_idx,idx[pre_idx]);
	     gsl_vector_set(graph[n_idx]->w,pre_idx,w[pre_idx]);
	   } 
	}
    }
  return graph;
}

int free_graph(presynaptic **graph, size_t n_neurons)
{
  for (size_t n_idx=0; n_idx<n_neurons; n_idx++)
    {
      if (graph[n_idx]->size > 0)
	{
	  gsl_vector_free(graph[n_idx]->w);
	  gsl_vector_uint_free(graph[n_idx]->idx);
	}
      free(graph[n_idx]);
      }
  free(graph);
  return 0;
}
