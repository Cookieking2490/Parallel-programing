#include <iostream>
#include <omp.h>
using namespace std;

#define GRAIN 1024

// Sequential sum function
long sequential_sum(int* A, int left, int right) {
	long sum = 0;
	for (int i = left; i < right; i++) {
		sum += A[i];
	}
	return sum;
}

// Parallel divide and conquer sum function
long parallel_sum(int* A, int left, int right) {
	if (right - left <= GRAIN) {
		return sequential_sum(A, left, right);
	}

	int mid = (left + right) / 2;

	long x = 0;
	long y = 0;

	#pragma omp taskgroup
	{
		#pragma omp task shared(x)
		{
			x = parallel_sum(A, left, mid);
		}

		y = parallel_sum(A, mid, right);
	}

	return x + y;
}

// Main function
int main() {
	int n = 1 << 20;
	int* A = new int[n];

	for (int i = 0; i < n; i++) {
		A[i] = 1;
	}

	//Sequential execution
	double seq_start = omp_get_wtime();
	long seq_result = sequential_sum(A, 0, n);
	double seq_end = omp_get_wtime();

	double seq_time = seq_end - seq_start;

	cout << "Sequential Time = " << seq_time << "\n\n";

	// thread counts
	int thread_counts[] = {2,4,8,16,32,64,128};
	for (int t : thread_counts) {

		omp_set_num_threads(t);

		long result;

		double start = omp_get_wtime();

		#pragma omp parallel
		{
			#pragma omp single
			{
				result = parallel_sum(A, 0, n);
			}
		}
		double end = omp_get_wtime();
		double parallel_time = end - start;

		cout << "Threads: " << t << endl;
		cout << "Parallel Time: " << parallel_time << endl;
		cout << "Speedup: " << seq_time / parallel_time << endl;
		cout << "_____________________________________\n";
	}

	delete[] A;
	return 0;
}