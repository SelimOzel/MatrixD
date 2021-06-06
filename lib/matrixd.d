module matrixd;

// D
import std.conv: to; 

// Enums
enum uint MAXROWS = 10000;
enum uint MAXCOLUMNS = 10000;

// Output is csv
string PrintMatrix(const Matrix matrix_IN) {
	string result;
	for(ulong r = 0; r < matrix_IN._nr; ++r) {
		for(ulong c = 0; c < matrix_IN._nc; ++c) {
			result ~= to!string(matrix_IN._m[r][c]);
			if(c == matrix_IN._nc - 1) result ~= "\n";
			else result ~= ",";
		}
	}
	return result;
}

// Lightweight
class Matrix {
public:
	// nxm filled with n
	this(
		const ulong rowLength_IN, 
		const ulong columnLength_IN, 
		const double n) pure {
		initialize(rowLength_IN, columnLength_IN, n);
	}

	// A = [[x1, x2 ...], [y1, y2, ...]]
    void opAssign(const double[][] matrixRHS_IN) pure {
		initialize(matrixRHS_IN.length, matrixRHS_IN[0].length, 0.0);
		for(ulong r = 0; r < _nr; ++r) {
			for(ulong c = 0; c < _nc; ++c) {
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
    	}    	
    }    

    // += scalar
    // -= scalar
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
    }      

    // matrix + matrix
    // matrix - matrix
    // matrix * matrix
    Matrix opBinary(string operation_IN)(const Matrix rhs_IN) pure const {
    	Matrix result = new Matrix(_nr, _nc, 0.0);
    	bool sum = operation_IN == "+";
    	bool subtract = operation_IN == "-";
    	bool multiply = operation_IN == "*";
    	ulong rhs_nr = rhs_IN._nr;
    	ulong rhs_nc = rhs_IN._nc;
    	if(sum || subtract){
	    	if(rhs_nr == _nr && rhs_nc == _nc) {
				for(ulong r = 0; r < _nr; ++r) {
					for(ulong c = 0; c < _nc; ++c) {
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
				for (ulong r = 0; r<_nr; r++) {
					for (ulong c = 0; c<rhs_nc; c++) {
						for (ulong k = 0; k<_nc; k++) {
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
    Matrix opBinary(string operation_IN)(const double rhs_IN) pure const {
    	Matrix result = new Matrix(_nr, _nc, 0.0);
    	if(operation_IN == "+"){	
			for(ulong r = 0; r < _nr; r++) {
				for(ulong c = 0; c < _nc; c++) {
					result._m[r][c] = _m[r][c] + rhs_IN;
				}
			}
    	}
    	else if(operation_IN == "-"){	
			for(ulong r = 0; r < _nr; r++) {
				for(ulong c = 0; c < _nc; c++) {
					result._m[r][c] = _m[r][c] - rhs_IN;
				}
			}
    	}    	
        return result;
    }        

/*
    // Get element
    //double opCall(ulong r, ulong c){
    //	return _m[][]
    //}
    double opIndexAssign(ulong r){
    	return 2.0;
    }
*/

    // Transpose
	Matrix T() pure const {
		Matrix result = new Matrix(_nc, _nr, 0.0);
		for (ulong r=0; r<result.Size()[0]; r++) {
			for (ulong c=0; c<result.Size()[1]; c++) {
				result._m[r][c] = _m[c][r];
			}
		}
		return result;
	}

    // Sums all elements
	double Sum() pure const {
		double result = 0.0;
		for(ulong r = 0; r < _nr; r++) {
			for(ulong c = 0; c < _nc; c++) {
				result += _m[r][c];
			}
		}
		return result;		
	}    

	// Sums all elements in row r.
	double Sum(const ulong r) pure const {
		if(r < _nr) {
			double result = 0.0;
			for(ulong c = 0; c < _nc; c++) {
				result += _m[r][c];
			}
			return result;	
		}
		else {
			string sum_row_err = "Sum row: index out of bounds";
			throw new Exception(sum_row_err);
		}
	}	

    // [rows, cols]
	ulong[2] Size() pure const {
		return [_nr, _nc];
	}    

private:
	void initialize(
		const ulong r_IN, 
		const ulong c_IN, 
		const double n_IN) pure {
		string initialize_err_maxrow = "initialize: Increase MAXROWS.";
		string initialize_err_maxcolumn = "initialize: Increase MAXCOLUMNS.";		
		if(r_IN > MAXROWS) throw new Exception(initialize_err_maxrow);
		if(c_IN > MAXCOLUMNS) throw new Exception(initialize_err_maxcolumn);
		if(r_IN != 0 && c_IN != 0) {
			for(ulong r = 0; r<r_IN; ++r) {
				for(ulong c = 0; c<c_IN; ++c) {
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

	ulong _nr = 0;
	ulong _nc = 0;
	double[MAXROWS][MAXCOLUMNS] _m;
}