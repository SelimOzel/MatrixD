// D
import std.conv: to;
import std.datetime.stopwatch: StopWatch, AutoStart;
import std.math: abs, PI, sin;
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

import matrixd.filters:
filter,
filtfilt;

// Third party
import plt = matplotlibd.pyplot;

void main() {
	writeln("Starting matrix tests ...");

	bool test_matrix = false;
	bool test_matrix_inverse = true;
	bool test_statistics = false; // just a bunch of thing I want to call easily
	bool test_filters = false;

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

		writeln("D Matrix Determinants");
		D.LU_Decomposition();
		Matrix[3] LU_D = D.LU_Decomposition();

		writeln(toCSV(D));
		writeln(toCSV(LU_D[0]));
		writeln(toCSV(LU_D[1]));	
		writeln(toCSV(LU_D[2]));

		writeln(D.Det_LU());
		writeln(D.Det(D.Size()[0]));
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
	if(test_matrix_inverse) {
		// Solve a linear equation
		// Solution: x vector is 
		// -0.41176470588,
		// -3.29411764706,
 		// 2.52941176471,
 		// 0.35294117647,
 		writeln("Ax=b");
		Matrix A_equations = new Matrix(
		[[1, 2, 3, 4],
		 [0, 3, 4, 5],
		 [1, 0, 2, 1],
		 [3, 4, 5, 2]]);

		Matrix B_equations = new Matrix(
		[[2],
		 [2],
		 [5],
		 [4]]);

		Matrix[3] A_equations_LU = A_equations.LU_Decomposition();	
		Matrix x_equations = A_equations.lup_solve(A_equations_LU[0], A_equations_LU[1], A_equations_LU[2], B_equations);
		
		writeln("A");
		writeln(toCSV(A_equations));
		writeln("A's lower");
		writeln(toCSV(A_equations_LU[0]));
		writeln("A's upper");
		writeln(toCSV(A_equations_LU[1]));	
		writeln("A's pivot");
		writeln(toCSV(A_equations_LU[2]));
		writeln("Ax = b Determinant: "~to!string(A_equations.Det(4)));
		writeln("Ax = b Determinant LUP: "~to!string(A_equations.Det_LU()));
		writeln("Result of A times x: ");		
		writeln(toCSV(A_equations*x_equations));
		assert(abs((A_equations*x_equations)[0,0] - B_equations[0,0]) <= float.epsilon);
		assert(abs((A_equations*x_equations)[1,0] - B_equations[1,0]) <= float.epsilon);
		assert(abs((A_equations*x_equations)[2,0] - B_equations[2,0]) <= float.epsilon);
		assert(abs((A_equations*x_equations)[3,0] - B_equations[3,0]) <= float.epsilon);

		writeln("LU Decomp-1");
		Matrix lu_decomp_1 = new Matrix(
		[[ 2.0, -1.0, -2.0],
		[ -4.0,  6.0,  3.0],
		[ -4.0, -2.0,  8.0]]);
		Matrix[3] LU_1 = lu_decomp_1.LU_Decomposition();
		writeln(toCSV(lu_decomp_1));
		writeln(toCSV(LU_1[0]));
		writeln(toCSV(LU_1[1]));	
		writeln(toCSV(LU_1[2]));
		writeln("LU Decomp-1: 3x3 Determinant: "~to!string(lu_decomp_1.Det(3)));
		writeln("LU Decomp-1: 3x3 Determinant LUP: "~to!string(lu_decomp_1.Det_LU()));	
		writeln();

		writeln("LU Decomp-2");
		Matrix lu_decomp_2 = new Matrix(
		[[ 1.0,  3.0,  5.0],
		[  2.0,  4.0,  7.0],
		[  1.0,  1.0,  0.0]]);
		Matrix[3] LU_2 = lu_decomp_2.LU_Decomposition();
		writeln(toCSV(lu_decomp_2));
		writeln(toCSV(LU_2[0]));
		writeln(toCSV(LU_2[1]));	
		writeln(toCSV(LU_2[2]));
		writeln("LU Decomp-2: 3x3 Determinant: "~to!string(lu_decomp_2.Det(3)));
		writeln("LU Decomp-2: 3x3 Determinant LUP: "~to!string(lu_decomp_2.Det_LU()));
		Matrix lu_decomp_2_result = lu_decomp_2.Inv()*lu_decomp_2;	
		Matrix lu_decomp_2_result_lu = lu_decomp_2.Inv_LU()*lu_decomp_2;
		writeln(lu_decomp_2_result.toCSV);	
		writeln();
		writeln(lu_decomp_2_result_lu.toCSV);
		writeln();

		writeln("LU Decomp-3");
		Matrix lu_decomp_3 = new Matrix(
		[[ 11.0,  9.0, 24.0,  2.0],
		[   1.0,  5.0,  2.0,  6.0],
		[   3.0, 17.0, 18.0,  1.0],
		[   2.0,  5.0,  7.0,  1.0]]);
		Matrix[3] LU_3 = lu_decomp_3.LU_Decomposition();
		writeln(toCSV(lu_decomp_3));
		writeln(toCSV(LU_3[0]));
		writeln(toCSV(LU_3[1]));
		writeln(toCSV(LU_3[2]));	
		writeln("LU Decomp-3: 4x4 Determinant: "~to!string(lu_decomp_3.Det(4)));
		writeln("LU Decomp-3: 4x4 Determinant LUP: "~to!string(lu_decomp_3.Det_LU()));	
		writeln();

		writeln("LU Decomp-4");
		Matrix lu_decomp_4 = new Matrix(
		[[ -5.0, -6.0, -3.0],
		[  -1.0,  0.0, -2.0],
		[  -3.0, -4.0, -7.0]]);
		Matrix[3] LU_4 = lu_decomp_4.LU_Decomposition();
		writeln(toCSV(lu_decomp_4));
		writeln(toCSV(LU_4[0]));
		writeln(toCSV(LU_4[1]));	
		writeln(toCSV(LU_4[2]));
		writeln("LU Decomp-4: 3x3 Determinant: "~to!string(lu_decomp_4.Det(3)));
		writeln("LU Decomp-4: 3x3 Determinant LUP: "~to!string(lu_decomp_4.Det_LU()));	
		writeln();						

		// 5x5 matrix performance
		auto myStopWatch = StopWatch(AutoStart.no);
		myStopWatch.start();
		Matrix Five = new Matrix(
		[[ 1, 13,  3, 12, -1],   
		[ 11,  2, -2,  4, 16],
		[  2,  0,  3,  3, 15],
		[ -5, -3,  5,  4, -3],
		[ 13, 15,  7, -4,  5]]);
		Matrix[3] LU_Five = Five.LU_Decomposition();
		Matrix Inv_5_by_5_result = Five.Inv_LU()*Five;
		writeln((Inv_5_by_5_result).toCSV);	

		myStopWatch.stop();
		writeln("5x5 Matrix constructor + determinant + inverse takes: "~to!string((to!double(myStopWatch.peek.total!"usecs")*0.000001))~" seconds");
		writeln();
		myStopWatch.reset();

		// 7x7 matrix performance
		myStopWatch = StopWatch(AutoStart.no);
		myStopWatch.start();
		Matrix Seven = new Matrix(
		[[ 12,  3,  8,  4, 1,  11,  3],   
		[   9,  4, -7,  4, 5,  -4,  9],
		[   1,  8,  5,  8, 0,  12, 13],
		[   9,  0,  1,  6, 3, -11,  2],
		[   0,  1,  2,  6, 2,   6,  0],
		[  10, -2,  0,  9, 4,   6, 14],
		[  28,  8,  3, 11, 7,  32, 15]]);
		Matrix[3] LU_Seven = Seven.LU_Decomposition();
		Matrix Inv_7_by_7_result = Seven.Inv_LU()*Seven;
		writeln((Inv_7_by_7_result).toCSV);

		myStopWatch.stop();
		writeln("7x7 Matrix constructor + determinant + inverse takes: "~to!string((to!double(myStopWatch.peek.total!"usecs")*0.000001))~" seconds");
		writeln();
		myStopWatch.reset();

		// 9x9 matrix performance
		myStopWatch = StopWatch(AutoStart.no);
		myStopWatch.start();
		Matrix Nine = new Matrix(
		[[ 2,  1,  8,  -5,  5,  2,   4, -13, 11],   
		[  6,  5,  6, -10,  7, 13,   3,  10,  9],
		[  5, 16, 19,   0, 12,  4,   4,   4,  8],
		[  4, 13,  0,  17,  2, 14,   1,   9,  7],
		[  7, -5, 12,  16,  3,  5,   6,   2,  8],
		[  6, 15, -3,   3,  9, -2, -12,   5,  2],
		[  3,  2,  6,   2, 11,  2,   3,   4, -9],
		[  8,  9, -7,   4, 13, 15,  18,  -8,  1],
		[  1,  0,  3,  14, 19, 12,   2,  11,  3]]);
		Matrix[3] LU_Nine = Nine.LU_Decomposition();
		Matrix Inv_9_by_9_result = Nine.Inv_LU()*Nine;
		writeln((Inv_9_by_9_result).toCSV);

		myStopWatch.stop();
		writeln("9x9 Matrix constructor + determinant + inverse takes: "~to!string((to!double(myStopWatch.peek.total!"usecs")*0.000001))~" seconds");
		writeln();
		myStopWatch.reset();

		// 10x10 matrix performance
		myStopWatch = StopWatch(AutoStart.no);
		myStopWatch.start();
		Matrix Ten = new Matrix(
		[[6.0, 6.0, 9.0, 5.0, 6.0, 4.0, 7.0, 4.0, 1.0, 2.0],   
		[ 2.0, 9.0, 6.0, 3.0, 8.0, 1.0, 5.0, 6.0, 3.0, 5.0],
		[ 2.0, 3.0, 3.0, 9.0, 1.0, 9.0, 9.0, 0.0, 5.0, 4.0],
		[ 4.0, 0.0, 6.0, 2.0, 4.0, 7.0, 4.0, 1.0, 3.0, 0.0],
		[ 7.0, 9.0, 4.0, 3.0, 2.0, 4.0, 4.0, 7.0, 0.0, 8.0],
		[ 4.0, 3.0, 7.0, 3.0, 2.0, 0.0, 7.0, 1.0, 5.0, 2.0],
		[ 3.0, 7.0, 9.0, 6.0, 1.0, 1.0, 7.0, 9.0, 4.0, 4.0],
		[ 9.0, 2.0, 2.0, 2.0, 9.0, 2.0, 2.0, 4.0, 5.0, 4.0],
		[ 6.0, 7.0, 2.0, 3.0, 9.0, 4.0, 2.0, 7.0, 7.0, 8.0],
		[ 8.0, 0.0, 8.0, 8.0, 4.0, 0.0, 5.0, 7.0, 1.0, 0.0]]);
		Matrix[3] LU_Ten = Ten.LU_Decomposition();
		Matrix Inv_10_by_10_result = Ten.Inv_LU()*Ten;
		writeln(Inv_10_by_10_result.toCSV);	

		myStopWatch.stop();
		writeln("10x10 Matrix constructor + determinant + inverse takes: "~to!string((to!double(myStopWatch.peek.total!"usecs")*0.000001))~" seconds");
		myStopWatch.reset();

		// 32x32 matrix performance
		myStopWatch = StopWatch(AutoStart.no);
		myStopWatch.start();
		Matrix TwirtyTwo = new Matrix( 
		[[8.1, 4.5, 0.7, 3.1, 2.0, 5.9, 5.2, 8.2, 4.4, 5.5, 8.3, 1.0, 1.7, 4.8, 8.8, 3.3, 5.6, 7.5, 7.8, 5.2, 5.7, 8.1, 6.0, 4.4, 3.6, 7.8, 1.7, 2.3, 5.6, 1.1, 6.3, 8.5],
		 [8.9, 8.9, 1.3, 7.7, 3.2, 5.5, 5.3, 7.0, 8.6, 2.4, 5.7, 1.5, 0.0, 1.5, 3.8, 8.4, 0.9, 1.2, 6.1, 3.3, 4.9, 1.8, 7.2, 4.5, 5.0, 0.9, 1.7, 5.2, 6.7, 5.8, 3.9, 9.0],
		 [1.7, 5.3, 3.7, 8.3, 3.2, 8.8, 6.5, 1.0, 1.0, 2.5, 2.9, 6.0, 6.5, 6.0, 5.5, 7.5, 0.6, 2.7, 5.7, 2.9, 7.4, 5.3, 7.0, 5.4, 2.7, 0.9, 8.5, 0.5, 4.4, 4.2, 3.1, 4.5],
		 [7.0, 6.1, 4.7, 1.8, 4.5, 1.5, 7.3, 4.1, 4.1, 4.3, 1.8, 7.0, 4.3, 2.8, 8.6, 0.1, 1.6, 6.7, 1.1, 0.5, 0.5, 0.4, 3.3, 3.6, 7.8, 1.9, 6.0, 0.3, 1.8, 6.9, 1.0, 8.6],
		 [6.9, 8.5, 0.4, 8.2, 5.1, 8.0, 0.6, 0.9, 2.5, 0.5, 7.7, 4.5, 1.4, 2.8, 5.7, 6.3, 9.0, 6.9, 3.7, 6.5, 7.1, 0.4, 4.1, 4.4, 4.1, 0.1, 2.5, 6.9, 2.5, 3.4, 0.2, 1.5],
		 [7.2, 0.2, 1.1, 4.2, 3.5, 4.7, 1.5, 2.0, 2.6, 7.3, 3.9, 7.6, 3.4, 1.9, 8.3, 1.8, 0.0, 2.3, 0.9, 5.4, 1.5, 5.9, 6.5, 5.7, 4.1, 2.0, 0.5, 6.0, 8.2, 0.9, 7.9, 5.8],
		 [1.6, 1.5, 1.0, 7.2, 7.7, 7.3, 8.4, 4.3, 1.8, 8.5, 8.4, 0.6, 4.6, 6.9, 3.5, 4.6, 5.3, 8.6, 5.8, 5.7, 7.8, 2.2, 4.8, 0.9, 3.9, 4.7, 7.1, 2.3, 7.6, 0.8, 8.4, 0.4],
		 [8.3, 6.2, 3.3, 8.6, 7.8, 3.5, 0.1, 8.1, 7.3, 5.7, 0.9, 1.6, 5.2, 7.3, 2.9, 1.1, 4.7, 1.8, 8.1, 6.8, 7.9, 2.6, 3.4, 5.7, 7.2, 6.1, 5.5, 5.6, 0.9, 8.8, 3.7, 0.6],
		 [2.4, 3.0, 1.2, 4.5, 8.3, 2.9, 7.9, 1.8, 6.1, 2.3, 1.8, 2.6, 2.1, 6.8, 6.9, 3.4, 1.5, 3.8, 3.8, 1.5, 5.0, 1.6, 6.5, 3.0, 6.9, 5.8, 2.0, 2.1, 5.3, 1.0, 7.3, 5.9],
		 [5.9, 3.5, 4.3, 6.1, 5.0, 2.5, 4.2, 6.3, 7.7, 7.4, 2.5, 4.4, 7.8, 7.2, 6.7, 6.9, 5.2, 1.3, 6.1, 3.1, 5.6, 1.8, 9.0, 5.7, 8.5, 8.6, 1.8, 3.0, 3.3, 3.7, 7.6, 8.6],
		 [7.8, 7.8, 4.1, 4.5, 6.4, 2.9, 3.0, 4.8, 0.7, 8.5, 0.6, 3.5, 9.0, 7.3, 7.7, 4.8, 8.7, 1.5, 4.6, 1.8, 2.6, 6.2, 8.6, 1.0, 2.4, 1.3, 3.9, 1.6, 8.7, 1.9, 2.1, 3.9],
		 [5.3, 6.7, 7.6, 1.8, 3.9, 4.3, 2.2, 0.6, 5.5, 6.2, 7.4, 6.6, 5.6, 8.5, 6.9, 3.9, 8.0, 0.5, 4.9, 7.3, 2.8, 6.1, 1.2, 4.3, 4.4, 8.6, 1.6, 1.8, 5.6, 4.1, 3.7, 5.1],
		 [1.7, 8.2, 8.3, 2.6, 5.7, 8.4, 2.0, 6.8, 6.1, 7.3, 1.2, 6.9, 1.3, 3.5, 4.8, 3.2, 5.6, 4.6, 1.6, 1.7, 8.9, 1.0, 3.1, 8.9, 5.6, 0.7, 7.3, 8.8, 2.6, 3.9, 3.8, 0.2],
		 [0.9, 1.1, 0.3, 6.2, 2.7, 4.2, 7.4, 7.8, 1.8, 1.0, 1.6, 6.5, 8.3, 8.3, 4.3, 1.2, 2.1, 2.3, 2.1, 8.7, 5.3, 8.9, 0.4, 5.2, 6.4, 8.8, 2.8, 2.2, 8.0, 5.9, 0.0, 2.2],
		 [8.5, 3.1, 1.9, 8.8, 1.3, 6.2, 0.5, 1.2, 6.2, 6.2, 6.8, 7.4, 5.5, 8.0, 6.9, 8.9, 5.9, 7.0, 8.4, 5.3, 3.1, 8.8, 0.4, 0.0, 7.9, 6.3, 3.5, 3.9, 4.6, 1.1, 7.5, 3.6],
		 [6.7, 0.3, 7.3, 3.8, 7.0, 7.8, 5.8, 2.9, 1.6, 5.1, 4.5, 8.8, 3.7, 1.4, 2.7, 1.3, 8.3, 3.4, 4.6, 1.3, 3.1, 6.0, 3.9, 3.4, 8.5, 4.2, 1.4, 3.5, 4.4, 8.6, 0.1, 6.5],
		 [1.1, 6.2, 7.0, 6.5, 2.0, 4.3, 4.7, 3.6, 1.6, 7.1, 2.1, 4.5, 3.6, 5.5, 4.4, 4.9, 3.1, 5.2, 1.4, 8.3, 7.5, 8.5, 1.9, 5.5, 4.3, 2.5, 6.4, 7.6, 0.9, 7.3, 2.5, 5.4],
		 [3.7, 6.2, 1.3, 7.6, 4.2, 1.3, 4.1, 8.9, 5.3, 5.4, 2.9, 0.7, 6.5, 2.6, 8.3, 2.4, 5.6, 6.0, 2.0, 7.9, 8.5, 1.9, 0.2, 7.6, 1.3, 0.3, 8.5, 8.5, 7.4, 1.4, 5.0, 5.6],
		 [3.9, 1.4, 7.1, 3.8, 8.9, 3.7, 0.8, 3.1, 5.7, 0.6, 8.0, 9.0, 6.9, 3.6, 3.9, 6.9, 7.9, 1.0, 1.6, 8.4, 2.1, 2.0, 5.7, 4.0, 4.6, 5.4, 2.2, 0.2, 0.6, 1.0, 0.2, 0.6],
		 [4.3, 6.8, 8.2, 6.0, 6.0, 2.9, 8.6, 4.6, 0.8, 2.4, 2.8, 6.9, 0.1, 5.2, 2.9, 8.4, 3.4, 1.2, 4.2, 0.7, 7.1, 0.1, 6.2, 0.9, 3.8, 8.3, 0.8, 2.3, 6.2, 5.3, 1.3, 1.8],
		 [4.1, 5.5, 2.3, 3.2, 3.4, 3.8, 4.0, 0.6, 7.3, 3.6, 8.6, 7.5, 0.2, 8.4, 3.6, 4.6, 3.5, 7.9, 3.6, 0.9, 4.4, 5.4, 7.2, 7.3, 4.1, 1.9, 7.3, 4.3, 7.7, 6.8, 6.4, 5.9],
		 [8.2, 0.2, 0.6, 1.9, 2.1, 3.9, 6.8, 5.0, 3.3, 4.0, 4.0, 5.6, 2.1, 3.4, 7.5, 3.9, 6.4, 1.2, 2.5, 4.5, 6.5, 7.6, 5.8, 8.2, 6.6, 7.3, 2.6, 3.3, 1.1, 7.1, 1.7, 4.5],
		 [6.0, 3.5, 1.8, 4.8, 1.0, 7.0, 1.6, 6.5, 3.7, 5.1, 5.8, 7.1, 4.2, 8.2, 4.5, 1.1, 4.3, 6.4, 6.0, 7.1, 6.2, 6.1, 7.5, 3.8, 6.0, 0.4, 3.9, 1.0, 0.8, 1.6, 4.2, 0.7],
		 [1.7, 2.1, 5.5, 8.3, 5.7, 4.9, 4.8, 8.7, 1.7, 4.4, 3.1, 0.6, 6.5, 7.5, 1.4, 0.7, 1.8, 0.3, 3.3, 0.4, 4.5, 3.8, 5.4, 3.2, 3.1, 4.1, 3.4, 3.3, 7.3, 1.9, 8.2, 7.5],
		 [8.1, 4.0, 3.1, 5.0, 1.5, 3.7, 7.4, 4.8, 0.2, 6.8, 0.3, 0.3, 2.1, 0.7, 2.4, 3.7, 7.1, 6.9, 3.0, 7.7, 6.7, 1.1, 4.0, 2.8, 5.4, 1.3, 4.2, 8.9, 5.8, 8.7, 5.9, 4.0],
		 [7.6, 4.5, 8.5, 5.9, 4.5, 7.5, 0.0, 0.9, 1.7, 0.2, 8.7, 4.6, 2.9, 8.4, 3.7, 0.4, 4.5, 7.0, 7.7, 0.7, 0.9, 7.7, 4.6, 7.9, 7.2, 3.4, 7.7, 7.1, 7.1, 0.3, 7.7, 6.8],
		 [8.9, 5.0, 5.8, 3.9, 7.7, 4.8, 2.5, 7.3, 0.6, 4.3, 1.1, 4.2, 5.2, 5.6, 4.1, 6.5, 0.5, 4.9, 6.5, 1.4, 0.6, 0.4, 1.7, 5.1, 1.6, 4.6, 4.5, 2.7, 4.7, 1.2, 2.5, 7.8],
		 [0.5, 7.1, 6.5, 0.9, 6.3, 4.6, 7.0, 6.4, 6.8, 4.7, 6.5, 0.4, 0.9, 6.5, 2.3, 5.9, 7.9, 3.9, 3.6, 3.6, 5.7, 8.1, 1.3, 3.4, 3.7, 1.6, 5.3, 2.9, 0.2, 6.2, 4.4, 2.4],
		 [0.6, 4.9, 1.7, 8.4, 2.6, 6.3, 7.4, 7.3, 4.6, 0.2, 1.5, 5.3, 6.9, 3.8, 6.0, 0.6, 0.7, 6.1, 4.8, 7.2, 5.2, 0.6, 6.4, 6.5, 0.0, 2.1, 2.9, 7.7, 0.4, 2.4, 2.2, 3.3],
		 [6.8, 7.2, 2.8, 3.5, 0.2, 5.9, 1.3, 8.0, 0.2, 8.6, 8.6, 1.9, 0.1, 1.8, 6.9, 5.2, 6.4, 1.7, 1.0, 3.6, 8.0, 0.2, 8.7, 3.2, 7.1, 5.8, 1.4, 5.0, 0.3, 7.1, 0.3, 2.4],
		 [5.7, 4.0, 4.1, 3.1, 7.1, 5.8, 2.5, 2.7, 8.9, 6.5, 7.7, 6.1, 6.4, 7.7, 5.4, 2.1, 0.8, 8.4, 7.5, 3.7, 7.4, 0.7, 4.4, 4.1, 1.6, 7.3, 5.0, 0.7, 2.0, 8.1, 8.5, 4.0],
		 [5.7, 6.3, 7.4, 6.7, 3.2, 1.9, 0.4, 8.4, 8.8, 7.1, 5.7, 2.4, 5.5, 1.6, 2.3, 7.6, 2.5, 7.0, 2.0, 5.4, 5.2, 4.0, 3.5, 2.7, 4.2, 5.7, 8.0, 3.8, 1.1, 2.4, 2.9, 6.4]]);			
		Matrix[3] LU_TwirtyTwo = TwirtyTwo.LU_Decomposition();
		Matrix Inv_32_by_32_result = TwirtyTwo.Inv_LU()*TwirtyTwo;
		writeln(Inv_32_by_32_result.toCSV);	

		myStopWatch.stop();
		writeln("32x32 Matrix constructor + determinant + inverse takes: "~to!string((to!double(myStopWatch.peek.total!"usecs")*0.000001))~" seconds");
		myStopWatch.reset();


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
		// Output of cpp hard coded for comparison
		Matrix y_filter_out_cpp = new Matrix([-0.01, -0.013721, 0.00881235, 0.0175505, 0.0324195, 0.0533212, 0.100134, 0.132712, 0.170887, 0.21447, 0.293248, 0.350712, 0.386633, 0.440792, 0.492974, 0.542974, 0.570594, 0.615647, 0.677955, 0.73735, 0.773678, 0.806793, 0.836567, 0.882882, 0.905633, 0.924731, 0.960101, 0.971682, 0.959428, 0.943308, 0.943308, 0.939428, 0.951682, 0.940102, 0.944732, 0.945633, 0.902882, 0.876567, 0.866793, 0.833677, 0.79735, 0.757955, 0.695647, 0.630594, 0.562974, 0.492974, 0.460792, 0.426633, 0.370711, 0.313248, 0.23447, 0.174608, 0.133899, 0.0925831, 0.0309016, -0.0309018, -0.112583, -0.193899, -0.254608, -0.31447, -0.353248, -0.410712, -0.486633, -0.540792, -0.592974, -0.622974, -0.650594, -0.675647, -0.737955, -0.75735, -0.813677, -0.826793, -0.836567, -0.882881, -0.885633, -0.904731, -0.920101, -0.951681, -0.939427, -0.943307, -0.943307, -0.939427, -0.931681, -0.9001, -0.88473, -0.865632, -0.862881, -0.836567, -0.826793, -0.813677, -0.757349, -0.717954, -0.695647, -0.670594, -0.642974, -0.592974, -0.520792, -0.466633, -0.410712, -0.333248, -0.27447]);
		Matrix y_filtfilt_out_cpp = new Matrix([-0.1, -0.0292282, 0.0413037, 0.107357, 0.172694, 0.235082, 0.296289, 0.35209, 0.406265, 0.4586, 0.510888, 0.558931, 0.604539, 0.649532, 0.693741, 0.735007, 0.773183, 0.812134, 0.847737, 0.875885, 0.89648, 0.913443, 0.926707, 0.938218, 0.94394, 0.94785, 0.94994, 0.944218, 0.934707, 0.925443, 0.91448, 0.899884, 0.881737, 0.856134, 0.825183, 0.787007, 0.741741, 0.697532, 0.652539, 0.602931, 0.550888, 0.4946, 0.436265, 0.38009, 0.326289, 0.273082, 0.220694, 0.163357, 0.101304, 0.0387717, -0.0240001, -0.0827719, -0.141304, -0.203357, -0.266694, -0.329082, -0.388289, -0.44209, -0.490265, -0.5386, -0.582888, -0.628931, -0.670539, -0.705532, -0.739741, -0.769007, -0.797183, -0.824133, -0.851737, -0.871884, -0.89048, -0.903443, -0.914706, -0.924218, -0.925939, -0.925849, -0.921939, -0.916217, -0.904706, -0.893443, -0.880479, -0.861884, -0.839736, -0.816133, -0.793182, -0.769007, -0.741741, -0.707532, -0.670539, -0.628931, -0.580888, -0.5326, -0.482265, -0.42409, -0.358289, -0.287082, -0.214694, -0.141357, -0.0633037, 0.0192282, 0.1]);

		// generate input signal
		Matrix X = new Matrix([0:100])*2.0*PI/100.0; // create x-axis ending at pi
		Matrix Y = sin(X); // full period sine wave
		Y = Y + noise(X, 0.0, 0.1); // add noise vector

		// create filter
		Matrix b_coeff = new Matrix([1./10., 1./10., 1./10., 1./10., 1./10.,1./10., 1./10., 1./10., 1./10., 1./10.]); 
		Matrix a_coeff = new Matrix([1.]); 
		Matrix zi = new Matrix([ 0. ]);

		// filter, filtfilt
		Matrix input_signal = Y; 
		Matrix y_filter_out_d; 
		Matrix y_filtfilt_d;
		y_filter_out_d = filter(b_coeff, a_coeff, input_signal, zi);
		
		y_filtfilt_d = filtfilt(b_coeff, a_coeff, input_signal);
/*
		// Fig - filter: input, cpp out, dlang out
		plt.plot(Y.toDouble_v, "b-");
		plt.plot(y_filter_out_cpp.toDouble_v, "r-");
		plt.plot(y_filter_out_d.toDouble_v, "g-");
		plt.xlabel("Samples");
		plt.ylabel("Magnitude");	
		plt.legend(["X", "Y_cpp", "Y_dlang"]);
		plt.grid();
		plt.savefig("filters_filter.png");
		plt.clear();	

		// Fig - filtfilt: input, cpp out, dlang out
		plt.plot(Y.toDouble_v, "b-");
		plt.plot(y_filtfilt_out_cpp.toDouble_v, "r-");
		plt.plot(y_filtfilt_d.toDouble_v, "g-");
		plt.xlabel("Samples");
		plt.ylabel("Magnitude");	
		plt.legend(["X", "Y_cpp", "Y_dlang"]);
		plt.grid();
		plt.savefig("filters_filtfilt.png");
		plt.clear();	
		*/
	}
}