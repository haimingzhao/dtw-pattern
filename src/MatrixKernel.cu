//
// Created by Mira on 22/08/2016.
//

#include "MatrixKernel.h"

#define getI_bl(i,j, nx, ny) ((nx-(i-j)-1) < ny) ? ( (nx-(i-j)-1)*(nx-(i-j))/2 + j ):( (ny-1)*(ny)/2 + (nx-(i-j)- ny)*ny + j ) ;
#define getI_ur(i,j, nx, ny, uj0) (ny<=nx) ? (uj0+(ny+ny-j+i+1)*(j-i)/2+i) : ( (j-i<=ny-nx)?(uj0+nx*(j-i)+i):(uj0+nx*(ny-nx)+(nx+ny-j+i+1)*(j-i-ny+nx)/2+i) );

