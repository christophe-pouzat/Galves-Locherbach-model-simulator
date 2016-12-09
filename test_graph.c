#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_uint.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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

int print_graph(presynaptic **graph, size_t n_neurons);

void print_usage_test_graph();

int read_args_test_graph(int argc, char ** argv,
			 size_t * n_neurons,
			 double * p_e, double * w_e_min, double * w_e_max,
			 double * p_i, double * w_i_min, double * w_i_max);

int main(int argc, char ** argv)
{
  size_t n_neurons;
  double p_e, w_e_min, w_e_max;
  double p_i, w_i_min, w_i_max;

  int status = read_args_test_graph(argc, argv, &n_neurons,
				    &p_e, &w_e_min, &w_e_max,
				    &p_i, &w_i_min, &w_i_max);

  if (status == -1) exit (EXIT_FAILURE);
  
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  presynaptic **graph=mk_graph(n_neurons,
			       p_e, w_e_min, w_e_max,
			       p_i, w_i_min, w_i_max,
			       r);
  gsl_rng_free (r);
  print_graph(graph, n_neurons);
  free_graph(graph, n_neurons);
  exit (EXIT_SUCCESS); 
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

int print_graph(presynaptic **graph, size_t n_neurons)
{
  for (size_t n_idx=0; n_idx<n_neurons; n_idx++)
    {
      fprintf(stdout,"Postsynaptic neuron: %d\n",(int) n_idx);
      if (graph[n_idx]->size > 0)
	{
	 for (size_t pre_idx=0; pre_idx<graph[n_idx]->size; pre_idx++)
	   {
	     fprintf(stdout,"\t W(%d -> %d) = %g\n",
		     (int) gsl_vector_uint_get(graph[n_idx]->idx,pre_idx),
		     (int) n_idx,
		     gsl_vector_get(graph[n_idx]->w,pre_idx)); 
	       
	   } 
	}
      fprintf(stdout,"\n");
    }
  return 0;
}

void print_usage_test_graph() {
  printf("Usage: \n"
	 "  --n_neurons <positive integer>: the number of neurons in the "
	 "network\n"
	 "  --p_e <double in (0,1)>: the probability of excitatory "
	 "connection between two neurons\n"
	 "  --w_e_min <positive double>: the minimal excitatory synaptic "
	 "weight\n"
	 "  --w_e_min <positive double>: the maximal excitatory synaptic "
	 "weight\n"
	 "  --p_i <double in (0,1)>: the probability of inhibitory "
	 "connection between two neurons\n"
	 "  --w_i_min <negative double>: the minimal inhibitory synaptic "
	 "weight\n"
	 "  --w_i_max <negative double>: the maximal inhibitory synaptic "
	 "weight\n"
	 "\n"
	 "The connection probalities are uniform. The synaptic weights "
	 "are drawn from\n"
	 "uniform distributions.\n"
	 "The rng seed can be set through the GSL_RNG_SEED environment "
	 "variable.\n"
	 "The rng type can be set through the GSL_RNG_TYPE environment "
	 "variable.\n");
}

int read_args_test_graph(int argc, char ** argv,
			 size_t * n_neurons,
			 double * p_e, double * w_e_min, double * w_e_max,
			 double * p_i, double * w_i_min, double * w_i_max)
{
  if (argc == 1) {
    print_usage_test_graph();
    return -1;
  }
  // Define default values
  *p_e = 0.1;
  *w_e_min = 0.2;
  *w_e_max = 0.4;
  *p_i = 0.2;
  *w_i_min = -0.1;
  *w_i_max = -0.05;
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
      {NULL,0,NULL,0}
    };
    int long_index =0;
    while ((opt = getopt_long(argc,argv,"hn:a:b:c:d:e:f:",long_options,\
			      &long_index)) != -1) {
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
      case 'h': print_usage_test_graph();
	return -1;
      default : print_usage_test_graph();
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
  return 0;
}
