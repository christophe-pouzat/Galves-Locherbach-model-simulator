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

#define W_E 1.0
#define W_I -4.0
#define VARPHI_0 0.005
#define VARPHI_K 1.0
#define N_NEURONS 2
#define TAU_E 10.0
#define D_E 1
#define TAU_I 25.0
#define D_I 5
#define NU_BAR 0.05
#define EARLY_STEPS 1000
#define TOTAL_STEPS 10000

typedef struct
{
  size_t size;
  gsl_vector_uint * idx;
  gsl_vector * w; 
} presynaptic;

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

int write_sim_2n_results(size_t n_neurons, GArray **history,
			 gsl_vector * u_0,
			 gsl_vector * u_1);

int write_sim_gl_preamble(sim_gl_params * params, gsl_rng * r);

int mk_one_step_2n(int t,
		   GArray **history,
		   presynaptic **graph,
		   sim_gl_params * params,
		   gsl_rng * r,
		   gsl_vector * u_0,
		   gsl_vector * u_1);


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

int main(void)
{
  sim_gl_params params={
    .n_neurons=N_NEURONS,.d_e=D_E, .d_i=D_I, .total_steps=TOTAL_STEPS,
    .early_steps=EARLY_STEPS, .p_e=1.0, .p_i=1.0, .w_e_max=W_E,
    .w_e_min=W_E, .w_i_max=W_I, .w_i_min=W_I, .varphi_0=VARPHI_0,
    .varphi_k=VARPHI_K, .tau_e=TAU_E, .tau_i=TAU_I, .nu_bar=NU_BAR,
    .G_e=G_e(TAU_E, D_E),.G_i=G_i(TAU_I, D_I) 
  };

  // Initialize RNG
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  // allocate memory for the network
  presynaptic **graph = malloc(N_NEURONS*sizeof(presynaptic*));
  graph[0] = malloc(sizeof(presynaptic));
  graph[0]->size=2;
  graph[0]->w = gsl_vector_alloc(2);
  graph[0]->idx = gsl_vector_uint_alloc(2);
  graph[1] = malloc(sizeof(presynaptic));
  graph[1]->size=1;
  graph[1]->w = gsl_vector_alloc(1);
  graph[1]->idx = gsl_vector_uint_alloc(1);
  // Initialize the network
  gsl_vector_uint_set(graph[0]->idx,0,1);
  gsl_vector_set(graph[0]->w,0,W_I);
  gsl_vector_uint_set(graph[0]->idx,1,1);
  gsl_vector_set(graph[0]->w,1,W_E);
  gsl_vector_uint_set(graph[1]->idx,0,0);
  gsl_vector_set(graph[1]->w,0,W_E);

  // allocate vectors containing u_0 and u_1
  gsl_vector * u_0 = gsl_vector_alloc(TOTAL_STEPS);
  gsl_vector * u_1 = gsl_vector_alloc(TOTAL_STEPS);

  // Allocate history
  GArray **history = malloc_garrays2 (params.n_neurons);

  // Write preamble
  write_sim_gl_preamble(&params, r);

  // Do the job
  for (int step_idx=0; step_idx < (int) TOTAL_STEPS; step_idx++)
    {
      mk_one_step_2n(step_idx, history, graph, &params,
		     r, u_0, u_1);
    }

  // Write results
  write_sim_2n_results(params.n_neurons, history, u_0, u_1);

  // Free memory taken up by history
  free_garrays2(history,params.n_neurons);

  gsl_vector_free(u_0);
  gsl_vector_free(u_1);
  gsl_rng_free (r);
  gsl_vector_free(params.G_e);
  gsl_vector_free(params.G_i);
  free_graph(graph, params.n_neurons);
  exit (EXIT_SUCCESS);
  
}

int mk_one_step_2n(int t,
		   GArray **history,
		   presynaptic **graph,
		   sim_gl_params * params,
		   gsl_rng * r,
		   gsl_vector * u_0,
		   gsl_vector * u_1)
{
  int res[N_NEURONS];
  double u;
  // Spike or no spike for each neuron
  if (t < EARLY_STEPS)
    {
      res[0] = spike_or_not_early(NU_BAR,r);
      gsl_vector_set(u_0,(size_t) t,0.0);
      res[1] = spike_or_not_early(NU_BAR,r);
      gsl_vector_set(u_1,(size_t) t,0.0);
    }
  else
    {
      // Neuron 0
      u = get_u_i(0, t, history, graph,
		  TAU_E, D_E, TAU_I, D_I);
      gsl_vector_set(u_0,(size_t) t,u);
      if (gsl_ran_flat(r,0.0,1.0) <= varphi(u, VARPHI_0, VARPHI_K))
	res[0]=1;
      else
	res[0]=0;
      // Neuron 1
      u = get_u_i(1, t, history, graph,
		  TAU_E, D_E, TAU_I, D_I);
      gsl_vector_set(u_1,(size_t) t,u);
      if (gsl_ran_flat(r,0.0,1.0) <= varphi(u, VARPHI_0, VARPHI_K))
	res[1]=1;
      else
	res[1]=0;
    }
  // Upate history
  for (size_t n_idx = 0; n_idx < N_NEURONS; n_idx++)
    {
      if (res[n_idx] == 1)
	g_array_append_val(history[n_idx],t);
    }
  return 0;
}

int write_sim_2n_results(size_t n_neurons, GArray **history,
			 gsl_vector * u_0,
			 gsl_vector * u_1)
{
  for (size_t n_idx = 0; n_idx < n_neurons; n_idx++)
    {
      fprintf(stdout,"# Start neuron %d with %d spikes\n",
	      (int) n_idx, history[n_idx]->len);
      for (size_t s_idx = 0; s_idx < history[n_idx]->len; s_idx++)
	{
	  fprintf(stdout,"%d\n", g_array_index(history[n_idx],
					       int,s_idx));
	}
      fprintf(stdout,"# End neuron %d\n",(int) n_idx);
      fprintf(stdout,"\n");
      fprintf(stdout,"\n");
    }
  fprintf(stdout,"# u_0\t u_1\n");
  for (size_t t_idx=0; t_idx < TOTAL_STEPS; t_idx++)
    fprintf(stdout,"%10.5g\t %10.5g\n",gsl_vector_get(u_0,t_idx),
	    gsl_vector_get(u_1,t_idx));
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
