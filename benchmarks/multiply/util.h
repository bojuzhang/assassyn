#ifndef __UTIL_H
#define __UTIL_H

void setStats(int enable);

static int verify(int n, const volatile int* test, const int* verify)
{
  int i;
  for (i = 0; i < n; i++)
  {
    if (test[i] != verify[i]) return i+1;
  }
  return 0;
}

static void printArray(const char name[], int n, const int arr[])
{
}

#endif
