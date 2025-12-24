#include "util.h"
#include "dataset1.h"

void vvadd( int n, int a[], int b[], int c[] )
{
  int i;
  for ( i = 0; i < n; i++ )
    c[i] = a[i] + b[i];
}

void setStats(int enable) {}

int main( int argc, char* argv[] )
{
  int results_data[DATA_SIZE];

  printArray( "input1", DATA_SIZE, input1_data );
  printArray( "input2", DATA_SIZE, input2_data );
  printArray( "verify", DATA_SIZE, verify_data );

  setStats(1);
  vvadd( DATA_SIZE, input1_data, input2_data, results_data );
  setStats(0);

  printArray( "results", DATA_SIZE, results_data );

  return verify( DATA_SIZE, results_data, verify_data );
}
