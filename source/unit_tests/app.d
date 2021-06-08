import std.math: sin, PI;
import std.stdio;
import matrixd;

void main() {
	writeln("Starting matrix tests ...");

	Matrix A = new Matrix(2, 2, 1.0); // 2x2 matrix with all elements 1.
	A = [[2,3],[3,2]]; // overload = 
	assert(A.Size()[0] == 2 && A.Size()[1] == 2);
	writeln(print(A));

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
	writeln(print(C));

	A = C; // Matrix assignment	
	assert(A.Size()[0] == 3 && A.Size()[1] == 2);	

	// Addition-Subtraction-Multiplication-Division(scalar)
	B = [[3,2],[2,3]];
	A = [[2,3],[3,2]]; // Set elements
	assert(A.Size()[0] == B.Size()[0] && A.Size()[1] == B.Size()[1]);	

	A += B; // cumulative matrix addition
	A += 1;	// cumulative scalar addition
	assert(A.Sum(0) + A.Sum(1) == A.Sum() && A.Sum() == 24);
	writeln(print(A));

	A -= 1; // cumulative matrix subtraction
	A = A - B; // matrix subtraction
	A = A - 1; // scalar subtraction
	assert(A.Sum(0) + A.Sum(1) == A.Sum() && A.Sum() == 6);

	A = [[3,2,1],[1,2,3],[0,1,2]]; // 3x3
	B = [[3,2],[1,4],[1,5]]; // 3x2
	A *= B;
	B = A.T();

	writeln(print(A));
	writeln(A[0,0]);
	writeln(A[0,1]);
	writeln(A.Sum(0));
	assert(A[0,0] + A[0,1] == A.Sum(0));
	assert(A[1,0] + A[1,1] == A.Sum(1));
	assert(A[2,0] + A[2,1] == A.Sum(2)); // row sum verifications
	assert(A.Sum() == A.Sum(0)+A.Sum(1)+A.Sum(2)); // row sum equals all element sum	

	writeln(print(B.T()));
	writeln(print(A));
	assert(B.T() == A);
	writeln(print(A));
	writeln(print(B[0])); // {12,8,3}
	C = B[0].T(); // Convert a row to column vector
	A = [[2,0,1],[0,2,1]]; 
	assert(A*B[0].T() == A*C);

	writeln(print(A*B[0].T())); // Convert B to column vector and multiply with 2x3 matrix. 

	// Determinant and inverse tests
	Matrix D = new Matrix(
	[[1.0, 0.0, 2.0, -1.0],   
	[ 3.0, 0.0, 0.0,  5.0],
	[ 2.0, 1.0, 4.0, -3.0],
	[ 1.0, 0.0, 5.0,  0.0]]);   
	assert(30 == D.Det(D.Size()[0]));
	writeln(print(D*D.Inv()));

	Matrix E = new Matrix([2:10]); // [2,3,4,5,6,7,8,9,10]
	Matrix F = new Matrix([2,3,4,5,6,7,8,9,10]);
	assert(E == F);
	writeln(print(E[0]));

	Matrix X = new Matrix([0:1000])*PI/1000.0; // create x-axis ending at pi
	Matrix Y = matrixd.sin(X); // full period sine wave
	writeln("Matrix tests passed!");
}