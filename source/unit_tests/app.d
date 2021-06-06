import std.stdio;
import matrix;

void main()
{
	writeln("Starting matrix tests ...");
}

/*
int main()
{
	std::cout<< "Linear Algebra Tests:\n\n";
	std::cout<< "Starting matrix tests...\n\n";

	// Assignment and creator operations
	Matrix A(2, 2, 1.0); // Initialize all elements to 1.
	A = {{2,3},{3,2}}; // Set elements
	assert(A.Size()[0] == 2 && A.Size()[1] == 2);

	Matrix::Print(A);
	std::cout<<"\n";

	A = {{7,8},{8,7},{8,9}}; // Set elements and reshape
	assert(A.Size()[0] == 3 && A.Size()[1] == 2);

	Matrix B(3, 2, 0.0); 
	B = {{3,2},{2,3},{2,1}}; // Set elements
	assert(B.Size()[0] == 3 && B.Size()[1] == 2);

	Matrix C(1, 1, 0.0); // Size can be easily changed later on
	C = B + A; // Matrix addition
	assert(C.Size()[0] == 3 && C.Size()[1] == 2);
	assert(C.Sum() == 60);
	assert(C.Sum(0) == 20);
	assert(C.Sum(1) == 20);
	assert(C.Sum(2) == 20);

	Matrix::Print(C);
	std::cout<<"\n";

	A = C; // Matrix assignment	
	assert(A.Size()[0] == 3 && A.Size()[1] == 2);

	// Addition-Subtraction-Multiplication-Division(scalar)
	B = {{3,2},{2,3}};
	A = {{2,3},{3,2}}; // Set elements
	assert(A.Size()[0] == B.Size()[0] && A.Size()[1] == B.Size()[1]);
	
	A += B; // cumulative matrix addition
	A += 1;	// cumulative scalar addition
	assert(A.Sum(0) + A.Sum(1) == A.Sum() && A.Sum() == 24);

	A -= 1; // cumulative matrix subtraction
	A = A - B; // matrix subtraction
	A = A - 1; // cumulative scalar subtraction
	assert(A.Sum(0) + A.Sum(1) == A.Sum() && A.Sum() == 6);

	A = {{3,2,1},{1,2,3},{0,1,2}}; // 3x3
	B = {{3,2},{1,4},{1,5}}; // 3x2
	A *= B;
	B = A.T();
	assert(A(0,0) + A(0,1) == A.Sum(0));
	assert(A(1,0) + A(1,1) == A.Sum(1));
	assert(A(2,0) + A(2,1) == A.Sum(2)); // row sum verifications
	assert(A.Sum() == A.Sum(0)+A.Sum(1)+A.Sum(2)); // row sum equals all element sum	
	assert(B.T() == A);

	Matrix::Print(A);
	std::cout<<"\n";	

	Matrix::Print(B[0].T()); // {12,8,3}
	std::cout<<"\n";

	C = B[0].T(); // Convert a row to column vector
	A = {{{2,0,1}},{{0,2,1}}}; 
	assert(A*B[0].T() == A*C);

	Matrix::Size(A);
	Matrix::Size(C);
	Matrix::Print(A*B[0].T()); // Convert B to column vector and multiply with 2x3 matrix. 
	std::cout<<"\n";

	Matrix state(3, 1, 0.0);
	state = std::vector<double> {2,2,2}; // I STRONGLY suggest using this convention WHEN declaring column vectors.
	Matrix::Size(state);
	C = A*state;

	Matrix D(std::vector<double> {2,2,2}); // Vector based constructor tests. 
	assert(state == D);
	assert (D.Size()[0] == 3 && D.Size()[1] == 1);
	assert(D(0,0) == 2 && D(1,0) == 2 && D(0,0) == 2);
	Matrix E(std::vector<std::vector<double>> {{2,0,1},{0,2,1}});
	assert(A == E);

	// Determinant and inverse tests
	E = {
			{1, 0, 2, -1},   
			{3, 0, 0, 5},   
			{2, 1, 4, -3},   
			{1, 0, 5, 0}   
	};   
	assert(30 == E.Det(E.Size()[0]));
	Matrix::Print(E*E.Inv());
	std::cout<<"\n";

	std::cout<< "Matrix tests passed!\n";
	return 1;
} 
*/