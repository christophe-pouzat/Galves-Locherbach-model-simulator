#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>

#define N_NEURONS 10

GArray ** malloc_garrays2(size_t n_neurons);

int free_garrays2(GArray ** history, size_t n_neurons);

int main(void)
{
  GArray **history = malloc_garrays2 (N_NEURONS);
  free_garrays2(history,N_NEURONS);
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
