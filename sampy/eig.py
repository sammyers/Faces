import numpy as np
from sympy import *
from sympy.matrices import *

def find_eigenvalues(matrix):
	"""Take a numpy square matrix and return its eigenvalues."""
	matrix = Matrix(matrix)
	n = matrix.shape[0]
	lamda = Symbol('lamda')
	solutions = solve(det(matrix - eye(n) * lamda), lamda)
	# return solutions
	return sorted([val.evalf() for val in solutions if val != 0], reverse=True)

def find_eigenvector(matrix, eigenvalue):
	"""Return the eigenvector corresponding to a given eigenvalue."""
	matrix = Matrix(matrix)
	n = matrix.shape[0]
	m = matrix - eye(n) * eigenvalue
	#WIP

def find_eigenvectors(matrix):
	"""Take a numpy square matrix and return the corresponding eigenvectors."""
	eigenvalues = find_eigenvalues(matrix)
	#WIP
