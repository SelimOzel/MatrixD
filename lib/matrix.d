module matrixd.matrix;

// D
import std.conv: to; 
import std.math: abs, pow, sin;
import std.random: dice;

// Enums
enum uint MAXROWS = 256;
enum uint MAXCOLUMNS = 257;

// Output is csv
string toCSV(const Matrix matrix_IN) {
	string result;
	for(int r = 0; r < matrix_IN._nr; ++r) {
		for(int c = 0; c < matrix_IN._nc; ++c) {
			result ~= to!string(matrix_IN._m[r][c]);
			if(c == matrix_IN._nc - 1) result ~= "\n";
			else result ~= ",";
		}
	}
	return result;
}

// vector output as double
double[] toDouble_v(const Matrix matrix_IN) {
	double[] matrix_double;
	for(int c = 0; c < matrix_IN._nc; ++c) {
		matrix_double ~= matrix_IN._m[0][c];
	}	
	return matrix_double;
}

// matrix output as double
double[][] toDouble_m(const Matrix matrix_IN) {
	double[][] matrix_double;
	for(int r = 0; r < matrix_IN._nr; ++r) {
		matrix_double ~= [[matrix_IN._m[r][0]]];
		for(int c = 1; c < matrix_IN._nc; ++c) {
			matrix_double[r] ~= matrix_IN._m[r][c];
		}
	}	
	return matrix_double;
}

// for each element in the matrix
Matrix sin(const Matrix matrix_IN) pure {
	Matrix sin_matrix = new Matrix(matrix_IN.Size()[0], matrix_IN.Size()[1], 0.0);
	for(int r = 0; r < matrix_IN._nr; ++r) {
		for(int c = 0; c < matrix_IN._nc; ++c) {
			sin_matrix[r,c] = sin(matrix_IN._m[r][c]);
		}
	}
	return sin_matrix;
}

// add random noise to each element
Matrix noise(const Matrix matrix_IN, double mean, double var) {
	Matrix noise_matrix = new Matrix(matrix_IN.Size()[0], matrix_IN.Size()[1], 0.0);
	for(int r = 0; r < matrix_IN._nr; ++r) {
		for(int c = 0; c < matrix_IN._nc; ++c) {
			noise_matrix[r,c] = mean + var*2.0*(dice(0.5, 0.5)-0.5);
		}
	}
	return noise_matrix;
}

// Lightweight
class Matrix {
public:
this() {
}

// nxm filled with n
this(
	const int rowLength_IN, 
	const int columnLength_IN, 
	const double n) 
pure {
	initialize(rowLength_IN, columnLength_IN, n);
}

// A = double[][]
this(const double[][] matrixRHS_IN) pure {
	initialize(to!int(matrixRHS_IN.length), to!int(matrixRHS_IN[0].length), 0.0);
	for(int r = 0; r < _nr; ++r) {
		for(int c = 0; c < _nc; ++c) {
			_m[r][c] = matrixRHS_IN[r][c];
		}
	}   	
}

// A = double[]
this(const double[] rowvectorRHS_IN) pure {
	initialize(1, to!int(rowvectorRHS_IN.length), 0.0);
	for(int r = 0; r < _nr; ++r) {
		for(int c = 0; c < _nc; ++c) {
			_m[r][c] = rowvectorRHS_IN[c];
		}
	}   	
}

// A = [1:10]
this(const int[int] x) pure {
	if(x.length == 1) {
		foreach(val; x.keys) {
			initialize(1, x[val]-val+1, 0.0);
			for(int r = 0; r < _nr; ++r) {
				for(int c = 0; c < _nc; ++c) {
					_m[r][c] = c+val;
				}
			}   			
		}
	}
	else {
		throw new Exception("Associative array length must be 1.");
	}
}

// A = [[x1, x2 ...], [y1, y2, ...]]
void opAssign(const double[][] matrixRHS_IN) pure {
	initialize(to!int(matrixRHS_IN.length), to!int(matrixRHS_IN[0].length), 0.0);
	for(int r = 0; r < _nr; ++r) {
		for(int c = 0; c < _nc; ++c) {
			_m[r][c] = matrixRHS_IN[r][c];
		}
	}    	
}		

// += matrix
// -= matrix
// *= matrix
void opOpAssign(string operation_IN)(const Matrix rhs_IN) pure {
	Matrix result = new Matrix(_nr, _nc, 0.0);
	if(operation_IN == "+"){	
		result = this + rhs_IN;	
		_m = result._m;		
	}
	else if(operation_IN == "-"){	
		result = this - rhs_IN;	
		_m = result._m;		
	}    	
	else if(operation_IN == "*"){	
		result = this * rhs_IN;	
		_m = result._m;
		_nr = result.Size()[0];
		_nc = result.Size()[1];
	}    	
}    

// += scalar
// -= scalar
// *= scalar
// /= scalar
void opOpAssign(string operation_IN)(const double rhs_IN) pure {
	Matrix result = new Matrix(_nr, _nc, 0.0);
	if(operation_IN == "+"){	
		result = this + rhs_IN;		
		_m = result._m;	
	}
	else if(operation_IN == "-"){	
		result = this - rhs_IN;		
		_m = result._m;	
	}
	else if(operation_IN == "*"){	
		result = this * rhs_IN;		
		_m = result._m;	
	}
	else if(operation_IN == "/"){	
		result = this / rhs_IN;		
		_m = result._m;	
	}  		 	
}      

// matrix + matrix
// matrix - matrix
// matrix * matrix
Matrix opBinary(string operation_IN)(const Matrix rhs_IN) pure const {
	Matrix result = new Matrix(_nr, _nc, 0.0);
	bool sum = operation_IN == "+";
	bool subtract = operation_IN == "-";
	bool multiply = operation_IN == "*";
	int rhs_nr = rhs_IN._nr;
	int rhs_nc = rhs_IN._nc;
	if(sum || subtract){
		if(rhs_nr == _nr && rhs_nc == _nc) {
			for(int r = 0; r < _nr; ++r) {
				for(int c = 0; c < _nc; ++c) {
					double rhs = rhs_IN._m[r][c];
					if(sum) {
						result._m[r][c] = _m[r][c] + rhs;
					}
					else if (subtract) {
						result._m[r][c] = _m[r][c] - rhs;
					}
				}
			}		
		}
		else {
			string matrix_addition_err = "matrix add/subtract: wrong dimensions.";
			throw new Exception(matrix_addition_err);
		}	  
	}  	
	else if(multiply) {			
		// Verify mXn * nXp condition
		if(_nc == rhs_nr) {
			result = new Matrix(_nr, rhs_nc, 0.0); // reshape to mXp
			for (int r = 0; r<_nr; r++) {
				for (int c = 0; c<rhs_nc; c++) {
					for (int k = 0; k<_nc; k++) {
						result._m[r][c] += _m[r][k] * rhs_IN._m[k][c];
					}
				}
			}				
		}
		else {
			string matrix_multip_err = "matrix multiplication: wrong dimensions.";
			throw new Exception(matrix_multip_err);
		}				
	}
	return result;
}    

// matrix + scalar
// matrix - scalar
// matrix * scalar
// matrix / scalar
Matrix opBinary(string operation_IN)(const double rhs_IN) pure const {
	Matrix result = new Matrix(_nr, _nc, 0.0);
	if(operation_IN == "+"){	
		for(int r = 0; r < _nr; r++) {
			for(int c = 0; c < _nc; c++) {
				result._m[r][c] = _m[r][c] + rhs_IN;
			}
		}
	}
	else if(operation_IN == "-"){	
		for(int r = 0; r < _nr; r++) {
			for(int c = 0; c < _nc; c++) {
				result._m[r][c] = _m[r][c] - rhs_IN;
			}
		}
	}   
	else if(operation_IN == "*"){	
		for(int r = 0; r < _nr; r++) {
			for(int c = 0; c < _nc; c++) {
				result._m[r][c] = _m[r][c] * rhs_IN;
			}
		}
	} 
	else if(operation_IN == "/"){	
		for(int r = 0; r < _nr; r++) {
			for(int c = 0; c < _nc; c++) {
				result._m[r][c] = _m[r][c] / rhs_IN;
			}
		}
	} 			 	
	return result;
}        

// A == B
override bool opEquals(Object o) pure const {
	auto rhs = cast(const Matrix)o;
	if(rhs.Size()[0] == Size()[0] && rhs.Size[1] == Size[1]){
		for(int r = 0; r<rhs._nr; ++r) {
			for(int c = 0; c<rhs._nc; ++c) {
				if(rhs[r,c] != _m[r][c]) return false;
			}
		}
	}
	else return false;
	return true;
}

// A[1,2] = x
void opIndexAssign(double val, int r, int c) pure {
	_m[r][c] = val;
}

// x = A[1,2]
double opIndex(int r, int c) pure const {
	if(r >= _nr) {
		string row_err = "opIndex[][]: row length error";
		throw new Exception(row_err);
	}
	if(c >= _nc) {
		string row_err = "opIndex[][]: column length error";
		throw new Exception(row_err);
	}	
	return _m[r][c];
}

// [x1, x2, ...] = A[1]
Matrix opIndex(int r) pure const {
	if(r >= _nr) {
		string row_err = "opIndex[]: row length error";
		throw new Exception(row_err);
	}	
	Matrix row_vector;
	if(r < _nr) {
		row_vector = new Matrix(1, _nc, 0.0);
		for (int c = 0; c<_nc; ++c) {
			row_vector[0, c] = _m[r][c];
		}
	}
	else throw new Exception("Can't access row.");
	return row_vector;
}

// Transpose
@safe
Matrix T() pure const {
	Matrix transpose = new Matrix(_nc, _nr, 0.0);
	for (int r=0; r<transpose.Size()[0]; r++) {
		for (int c=0; c<transpose.Size()[1]; c++) {
			transpose._m[r][c] = _m[c][r];
		}
	}
	return transpose;
}

// Lower upper triangle decomposition
@safe
Matrix[3] LU_Decomposition() pure const {
	Matrix[3] result;

	int nr = Size()[0];
	int nc = Size()[1];

	Matrix lower = new Matrix(nr, nc, 0.0);
	Matrix upper = new Matrix(nr, nc, 0.0);
	Matrix pivot = new Matrix(nr, nr, 0.0);

	if(nr == nc) {	
    	Matrix perm = new Matrix([0:nc]);     	
    	double[MAXCOLUMNS][MAXROWS] input1 = _m;
	    for (int j = 0; j < nr; ++j) {
	        int max_index = j;
	        double max_value = 0;
	        for (int i = j; i < nr; ++i) {
	            double value = abs(input1[to!int(perm[0, i])][j]);
	            if (value > max_value) {
	                max_index = i;
	                max_value = value;
	            }
	        }
	        if (max_value <= float.epsilon)
	            throw new Exception("matrix is singular");
	        if (j != max_index) {
	        	double dummy = perm[0,j];
	        	perm[0, j] = perm[0, max_index];
	        	perm[0, max_index] = dummy;
	        }
	        int jj = to!int(perm[0, j]);
	        for (int i = j + 1; i < nr; ++i) {
	            int ii = to!int(perm[0, i]);
	            input1[ii][j] /= input1[jj][j];
	            for (int k = j + 1; k < nr; ++k)
	                input1[ii][k] -= input1[ii][j] * input1[jj][k];
	        }
	    }
	    for (int j = 0; j < nr; ++j) {
	        lower[j, j] = 1;
	        for (int i = j + 1; i < nr; ++i)
	            lower[i, j] = input1[to!int(perm[0, i])][j];
	        for (int i = 0; i <= j; ++i)
	            upper[i, j] = input1[to!int(perm[0, i])][j];
	    }
    	for (int i = 0; i < nr; ++i)
        	pivot[i, to!int(perm[0,i])] = 1.0;
	}
	else {
		throw new Exception("LU Decomposition error: not square\n");
	}  	  

	result[0] = lower;
	result[1] = upper;
	result[2] = pivot;
	return result;
}

@safe
Matrix Inv() pure const {
	int nr = Size()[0];
	int nc = Size()[1];

	if(nr == nc) {	
	    // Find determinant of A[][] 
	    double determinant = Det(nr); 
	    if (determinant == 0) { 
	        throw new Exception("Inverse error: determinant must be non-zero\n"); 
	    } 
	  
	    // Inverse
	    Matrix inverse = new Matrix(nr,nr,0.0);

	    // Find adjoint 
	    Matrix adj = new Matrix(nr, nr, 0.0);
	    adjoint(adj); 

	    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
	    for (int i=0; i<nr; i++) 
	        for (int j=0; j<nr; j++) 
	        	inverse[i,j] = adj[i,j]/determinant;
	  
	    return inverse; 
    }	
	else {
		throw new Exception("Inverse error: not square\n");
	}    
}

@safe
Matrix Inv_LU() pure const {
	int nr = Size()[0];
	int nc = Size()[1];

	if(nr == nc) {	
	    // Find determinant of A[][] 
	    double determinant = Det_LU(); 
	    if (abs(determinant) <= float.epsilon) { 
	        throw new Exception("Inverse error: determinant must be non-zero\n"); 
	    } 
		// decompose into lower, upper triangular and obtain pivot matrix
		Matrix[3] LU = LU_Decomposition();	  

		// identity columns
		Matrix e = Identity(nr); 

	    // x are columns of the inverse
	    Matrix inverse = new Matrix(nr,nr,0.0);	    
	    Matrix x;
	    for (int i = 0; i<nr; ++i) {
		    x = lup_solve(LU[0], LU[1], LU[2], e[i].T);
		    for(int j = 0; j<nr; ++j) {
		    	inverse[j,i] = x[j,0];
		    }
	    }
	    return inverse;
    }	
	else {
		throw new Exception("Inverse error: not square\n");
	}    
}

// x = lup_solve(L, U, P, b) is the solution to L U x = P b
// L must be a lower-triangular matrix
// U must be an upper-triangular matrix of the same shape as L
// P must be a permutation matrix of the same shape as L
// b must be a vector of the same leading dimension as L
@safe
Matrix lup_solve(Matrix L, Matrix U, Matrix P, Matrix b) pure const {
    Matrix z = P*b;
    Matrix x = lu_solve(L, U, z);
    return x;
}

@safe
double Det_LU() pure const {
	Matrix[3] LU = LU_Decomposition();
	int nr = Size()[0];
	double det = 1.0;
	Matrix P = LU[2];
	double nswaps = to!double(P.Diag().Size()[0]) - P.Diag().Sum() - 1.0;
	for(int i = 0; i< nr; ++i) {
		det *= LU[1][i,i];
	}
	return det*pow(-1,nswaps);
}

// Obtained from geeks-for-geeks: https://www.geeksforgeeks.org/determinant-of-a-matrix/
@safe
double Det(int n) pure const {
	int nr = Size()[0];
	int nc = Size()[1];

	if(nr == nc) {
		//  Base case : if matrix contains single element 
		double D = 0; // Initialize result 
		if (n == 1) return _m[0][0]; 

		//std::vector<std::vector<double>> temp(n, std::vector<double>(n)); // To store cofactors 
		Matrix temp = new Matrix(n,n,0.0); // To store cofactors 

		int sign = 1;  // To store sign multiplier 

		// Iterate for each element of first row 
		for (int f = 0; f < n; f++) {
			// Getting Cofactor of mat[0][f] 
			cofactor(temp, 0, f, n); 
			D += sign * _m[0][f] * temp.Det(n - 1); 
			// terms are to be added with alternate sign 
			sign = -sign; 
		}

		return D;	
	}
	else {
		throw new Exception("Determinant error: not square\n");
	}
}

// Sums all elements
@safe
double Sum() pure const {
	double all_sum = 0.0;
	for(int r = 0; r < _nr; r++) {
		for(int c = 0; c < _nc; c++) {
			all_sum += _m[r][c];
		}
	}
	return all_sum;		
}    

// Sums all elements in row r.
@safe
double Sum(const int r) pure const {
	if(r < _nr) {
		double row_sum = 0.0;
		for(int c = 0; c < _nc; c++) {
			row_sum += _m[r][c];
		}
		return row_sum;	
	}
	else {
		string sum_row_err = "Sum row: index out of bounds";
		throw new Exception(sum_row_err);
	}
}	

// returns a matrix containing all diagonals
@safe
Matrix Diag() pure const {
	if(_nr != _nc) {
		string diag_square_err = "Diag: not a square";
		throw new Exception(diag_square_err);		
	}
	Matrix result = new Matrix(1, _nr, 0.0);
	for(int i = 0; i<_nr; ++i){
		result[0, i] = _m[i][i];
	}
	return result;
}

// create nxn identity
@safe
Matrix Identity(int n) pure const {
	Matrix result = new Matrix(n, n, 0.0);
	for(int i = 0; i<n; ++i) {
		result[i,i] = 1.0;
	}
	return result;
}

// [rows, cols]
@safe
int[2] Size() pure const {
	return [_nr, _nc];
}    

// 0 rows, 0 cols is empty
@safe
bool empty() pure const {
	if(Size()[0] == 0 && Size()[1] == 0) return true;
	else return false;
}    

private:
void initialize(
	const int r_IN, 
	const int c_IN, 
	const double n_IN) 
pure {
	string initialize_err_maxrow = "initialize: Increase MAXROWS.";
	string initialize_err_maxcolumn = "initialize: Increase MAXCOLUMNS.";		
	if(r_IN > MAXROWS) throw new Exception(initialize_err_maxrow);
	if(c_IN > MAXCOLUMNS) throw new Exception(initialize_err_maxcolumn);
	if(r_IN != 0 && c_IN != 0) {
		for(int r = 0; r<r_IN; ++r) {
			for(int c = 0; c<c_IN; ++c) {
				_m[r][c] = n_IN;
			}
		}
		_nr = r_IN;
		_nc = c_IN;
	}
	else {
		throw new Exception("Matrix dimensions can't be zero.");
	}
}

// Obtained from geeks-for-geeks: https://www.geeksforgeeks.org/determinant-of-a-matrix/
@safe
void cofactor(ref Matrix temp, int p, int q, int n) pure const {
    int i = 0, j = 0; 
  
    // Looping for each element of the matrix 
    for (int row = 0; row < n; row++) { 
        for (int col = 0; col < n; col++) { 
            //  Copying into temporary matrix only those element 
            //  which are not in given row and column 
            if (row != p && col != q) { 
            	//temp.Set(i,j++,_m[row][col]);
                temp[i,j++] = _m[row][col]; 
  
                // Row is filled, so increase row index and 
                // reset col index 
                if (j == n - 1) { 
                    j = 0; 
                    i++; 
                } 
            } 
        } 
    } 	
}

// x = forward_sub(L, b) is the solution to L x = b
// L must be a lower-triangular matrix
// b must be a vector of the same leading dimension as L
@safe
Matrix forward_sub(Matrix L, Matrix b) pure const {
    int nr = L.Size()[0];
    Matrix x = new Matrix(nr, 1, 0.0);
    for (int i = 0; i<nr; ++i) {
        double tmp = b[i,0];
        for (int j = 0; j<i; ++j){
            tmp -= L[i,j] * x[j,0];
        }
        x[i,0] = tmp / L[i,i];
    }
    return x;
}

// x = back_sub(U, b) is the solution to U x = b
// U must be an upper-triangular matrix
// b must be a vector of the same leading dimension as U
@safe
Matrix back_sub(Matrix U, Matrix b) pure const {
    int nr = U.Size()[0];
    Matrix x = new Matrix(nr, 1, 0.0);
    for (int i = nr-1; i>-1; --i) {
        double tmp = b[i,0];
        for (int j = i+1; j<nr; ++j){
            tmp -= U[i,j] * x[j,0];
    	}
        x[i,0] = tmp / U[i,i];
    }
    return x;    
}

// x = lu_solve(L, U, b) is the solution to L U x = b
// L must be a lower-triangular matrix
// U must be an upper-triangular matrix of the same size as L
// b must be a vector of the same leading dimension as L
@safe
Matrix lu_solve(Matrix L, Matrix U, Matrix b) pure const {	
    Matrix y = forward_sub(L, b);
    Matrix x = back_sub(U, y);
    return x;
}

// Obtained from geeks-for-geeks: https://www.geeksforgeeks.org/adjoint-inverse-matrix/
@safe
void adjoint(ref Matrix adj) pure const { 
	int N = _nr;
    if (N == 1) { 
        adj[0,0] = 1.0; 
        return; 
    } 
  
    // temp is used to store cofactors of A[][] 
    int sign = 1;
    //std::vector<std::vector<double>> temp(N, std::vector<double>(N)); // To store cofactors 
  	Matrix temp = new Matrix(N, N, 0.0);

    for (int i=0; i<N; i++) { 
        for (int j=0; j<N; j++) { 
            // Get cofactor of A[i][j] 
            cofactor(temp, i, j, N); 

            // sign of adj[j][i] positive if sum of row 
            // and column indexes is even. 
            sign = ((i+j)%2==0)? 1: -1; 
  
            // Interchanging rows and columns to get the 
            // transpose of the cofactor matrix 
            adj[j,i] = (sign)*(temp.Det(N-1));
        } 
    } 
} 

int _nr = 0;
int _nc = 0;
double[MAXCOLUMNS][MAXROWS] _m;
}