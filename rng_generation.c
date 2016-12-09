#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define MIN -10.0
#define MAX 10.0
#define NB 1000000000

int main()
{
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  for (size_t i=0; i<NB; i++)
    {
      double u=gsl_ran_flat(r,MIN,MAX);
    }
  exit (EXIT_SUCCESS);
}
