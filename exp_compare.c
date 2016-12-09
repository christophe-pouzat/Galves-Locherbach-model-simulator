#include <stdio.h>
#include <stdlib.h>

#define FAST_EXP
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

int main()
{
  for (size_t i=0; i < 1002; i++)
    {
      double x = i*0.01-5;
      fprintf(stdout,"%7.5g\t%12.10g\t%12.10g\n",x,1/(1+exp(-x)),
	      1/(1+EXP(-x)));
    }
  exit (EXIT_SUCCESS);
}
