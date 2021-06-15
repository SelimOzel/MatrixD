// D
import std.math: sin, PI;
import std.stdio;

// MatrixD
import matrixd.matrix: 
Matrix, 
noise, 
sin, 
toCSV, 
toDouble_m, 
toDouble_v;

import matrixd.statistics:  
compute_correlation,
compute_mean, 
compute_ordinary_least_squares,
compute_std, 
generate_autoregressive_process;

// Third party
import plt = matplotlibd.pyplot;

void main() {
	writeln("Starting matrix tests ...");

	bool test_matrix = true;
	bool test_statistics = true; // just a bunch of thing I want to call easily
	bool test_filters = true;

	if(test_matrix) {
		Matrix A = new Matrix(2, 2, 1.0); // 2x2 matrix with all elements 1.
		A = [[2,3],[3,2]]; // overload = 
		assert(A.Size()[0] == 2 && A.Size()[1] == 2);
		writeln(toCSV(A));

		A = [[7,8],[8,7],[8,9]]; // change dimensions
		assert(A.Size()[0] == 3 && A.Size()[1] == 2); // assert the dimension change

		Matrix B = new Matrix(3, 2, 0.0); // second matrix
		B = [[3,2],[2,3],[2,1]]; // keeps the dimension same. only sets elements
		assert(B.Size()[0] == 3 && B.Size()[1] == 2);	

		Matrix C = new Matrix(1, 1, 0.0); // Size change automatically in the next line
		C = B + A; // Matrix addition
		assert(C.Size()[0] == 3 && C.Size()[1] == 2);
		assert(C.Sum() == 60);
		assert(C.Sum(0) == 20);
		assert(C.Sum(1) == 20);
		assert(C.Sum(2) == 20);	
		writeln(toCSV(C));

		A = C; // Matrix assignment	
		assert(A.Size()[0] == 3 && A.Size()[1] == 2);	

		// Addition-Subtraction-Multiplication-Division(scalar)
		B = [[3,2],[2,3]];
		A = [[2,3],[3,2]]; // Set elements
		assert(A.Size()[0] == B.Size()[0] && A.Size()[1] == B.Size()[1]);	

		A += B; // cumulative matrix addition
		A += 1;	// cumulative scalar addition
		assert(A.Sum(0) + A.Sum(1) == A.Sum() && A.Sum() == 24);
		writeln(toCSV(A));

		A -= 1; // cumulative matrix subtraction
		A = A - B; // matrix subtraction
		A = A - 1; // scalar subtraction
		assert(A.Sum(0) + A.Sum(1) == A.Sum() && A.Sum() == 6);

		A = [[3,2,1],[1,2,3],[0,1,2]]; // 3x3
		B = [[3,2],[1,4],[1,5]]; // 3x2
		A *= B;
		B = A.T();

		writeln(toCSV(A));
		writeln(A[0,0]);
		writeln(A[0,1]);
		writeln(A.Sum(0));
		assert(A[0,0] + A[0,1] == A.Sum(0));
		assert(A[1,0] + A[1,1] == A.Sum(1));
		assert(A[2,0] + A[2,1] == A.Sum(2)); // row sum verifications
		assert(A.Sum() == A.Sum(0)+A.Sum(1)+A.Sum(2)); // row sum equals all element sum	

		writeln(toCSV(B.T()));
		writeln(toCSV(A));
		assert(B.T() == A);
		writeln(toCSV(A));
		writeln(toCSV(B[0])); // {12,8,3}
		C = B[0].T(); // Convert a row to column vector
		A = [[2,0,1],[0,2,1]]; 
		assert(A*B[0].T() == A*C);

		writeln(toCSV(A*B[0].T())); // Convert B to column vector and multiply with 2x3 matrix. 

		// Determinant and inverse tests
		Matrix D = new Matrix(
		[[1.0, 0.0, 2.0, -1.0],   
		[ 3.0, 0.0, 0.0,  5.0],
		[ 2.0, 1.0, 4.0, -3.0],
		[ 1.0, 0.0, 5.0,  0.0]]);   
		assert(30 == D.Det(D.Size()[0]));
		writeln(toCSV(D*D.Inv()));

		writeln(toCSV(A)); // check row/cols are matching
		writeln(toDouble_m(A));
		assert(toDouble_m(A)[0] == toDouble_v(A[0]));
		assert(toDouble_m(A)[1] == toDouble_v(A[1]));

		Matrix E = new Matrix([2:10]); // [2,3,4,5,6,7,8,9,10]
		Matrix F = new Matrix([2,3,4,5,6,7,8,9,10]);
		assert(E == F);
		writeln(toCSV(E[0]));

		Matrix X = new Matrix([0:1000])*PI/1000.0; // create x-axis ending at pi
		Matrix Y = sin(X); // full period sine wave
		Y = Y + noise(X, 0.0, 0.1); // add noise vector

		assert(toDouble_v(Y).length == toDouble_v(X).length);

		Matrix G = new Matrix();
		assert(G.empty() == true);
		G = [[2,4],[4,8],[12,24]];
		G /= 2.0;
		assert(G[0,1] == 2);
		G[1,1] = G[1,1] / 2.0;
		assert(G[1, 1] == 2);
		writeln("Matrix tests passed!");
	}

	if(test_statistics) {
		double[] time_series_1;
		double[] time_series_2;

		int n = 1000;
		double[] t;
		for (int i = 0; i<n; i++) {t ~= i;}

		// Stationary process
		time_series_1 = generate_autoregressive_process(1.0, 0.95, 1.0, n);
		time_series_2 = generate_autoregressive_process(1.0, 0.95, 1.0, n);
		writeln(compute_mean(time_series_1));
		writeln(compute_std(time_series_1));
		writeln(compute_correlation(time_series_1, time_series_2));
		writeln();
		plt.plot(t, time_series_1, "r-", ["label": "$rho<1$"]);
		plt.plot(time_series_2, "b-", ["label": "$rho<1$"]);
		plt.legend();
		plt.savefig("statistics_stationary_process.png");
		plt.clear();

		// Non-Stationary process
		time_series_1 = generate_autoregressive_process(0.0, 1.0, 1.0, n);
		time_series_2 = generate_autoregressive_process(0.0, 1.0, 1.0, n);
		writeln(compute_mean(time_series_1));
		writeln(compute_std(time_series_1));
		writeln(compute_correlation(time_series_1, time_series_2));
		writeln(compute_ordinary_least_squares(time_series_1, time_series_2));
		writeln();
		plt.plot(time_series_1, "r-", ["label": "$rho=1$"]);
		plt.plot(time_series_2, "b-", ["label": "$rho=1$"]);
		plt.legend();
		plt.savefig("statistics_non_stationary_process.png");
		plt.clear();

		double[] time_series_3;
		double[] time_series_4;

		// Stationary process
		time_series_3 = generate_autoregressive_process(0.0, 0.9, 1.0, 1000);
		time_series_4 = generate_autoregressive_process(0.0, 0.9, 1.0, 1000);
		plt.plot(time_series_3, time_series_4, "r-", ["label": "$rho<1$"]);
		plt.legend();
		plt.savefig("statistics_ugly_phases.png");
		plt.clear();

		// Non-Stationary process
		time_series_3 = generate_autoregressive_process(0.0, 1.0, 1.0, 1000);
		time_series_4 = generate_autoregressive_process(0.0, 1.0, 1.0, 1000);
		plt.plot(time_series_3, time_series_4, "r-", ["label": "$rho=1$"]);
		plt.legend();
		plt.savefig("statistics_nice_phases.png");
		plt.clear();		

		double[] time_series;
		double[] time_series_estimated;
		double[2] ols;

		int l = 1000;
		double[] t2;
		for (int i = 0; i<l; i++) {t2 ~= i;}

		time_series = generate_autoregressive_process(0.1, 1.0, 1.0, l);
		ols = compute_ordinary_least_squares(t2, time_series);
		for (int i = 0; i<l; i++) {time_series_estimated ~= ols[0] + ols[1]*i;}

		plt.plot(t2, time_series, "o", ["markersize": 0.5]);
		plt.plot(t2, time_series_estimated, "r-");
		plt.legend();
		plt.grid();
		plt.savefig("statistics_ordinary_least_squares.png");
		plt.clear();		
	}

	if(test_filters) {

	}
}