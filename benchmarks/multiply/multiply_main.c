#include "util.h"
#include "multiply.h"
#include "dataset1.h"

void setStats(int enable) {}

int main( int argc, char* argv[] )
{
  int i;
  int results_data[DATA_SIZE];

  printArray( "input1", DATA_SIZE, input_data1 );
  printArray( "input2", DATA_SIZE, input_data2 );
  printArray( "verify", DATA_SIZE, verify_data );

  setStats(1);
  for (i = 0; i < DATA_SIZE; i++)
  {
    results_data[i] = multiply( input_data1[i], input_data2[i] );
  }
  setStats(0);

  printArray( "results", DATA_SIZE, results_data );

  return verify( DATA_SIZE, results_data, verify_data );
}
