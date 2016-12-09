#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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
#if defined(FAST_EXP)
      EXP(u);
#else
      exp(u);
#endif
    }
  exit (EXIT_SUCCESS);
}
