#include<bits/stdc++.h>
using namespace std;
typedef vector<int> VI;
typedef vector<VI> VVI;
int BG1[46][68]={0};
const int Z=15;
const int Zmax=383;
int mat[Z][Z]={0};
int H[46*Z][68*Z]={0}; // (n-k)*(k+(n-k))

void print(FILE* file,VVI& A){
	int n=A.size();
	for( int i=0;i<n;i++)
	{
		int n_=A[i].size();
		for( int j=0;j<n_;j++)
			fprintf(file,"%d ",A[i][j]);
		fprintf(file,"\n");
	}
}
void I(int shifts,int Z){
	for( int i=0;i<Z;i++)
		for( int j=0;j<Z;j++)
			mat[i][j]=0;
	if(shifts<0)return;
	for( int i=0;i<Z;i++)
		mat[i][(i+shifts)%Z]=1;
}
int main(){
	for( int i=0;i<46;i++)
	{
		for( int j=26;j<68;j++)
		{
			if(i<4)BG1[i][j]=-1;
			else if(22+i!=j)BG1[i][j]=-1;
		}
	}
	for( int i=0;i<46;i++)
	{
		for( int j=0;j<26;j++)
		{
			if(i<4 && j>=22)continue;
			BG1[i][j]=rand()%Zmax;
		}
	}
	// k = 22,22*2,22*3... Depending on the value of Z.
	// => (n-k) = 46,46*2,46*3... Depending on the value of Z.
	// You can use puncturing to reduce the number of parity bits.
	/*
		| A | E | O |
		--------------
		| B | C | I | 
	*/
	//We have to design the E part of the matrix careful.
	BG1[0][22]=0;BG1[0][23]=-1;BG1[0][24]=-1;BG1[0][25]=-1;
	BG1[1][22]=1;BG1[1][23]=0;BG1[1][24]=-1;BG1[1][25]=-1;
	BG1[2][22]=-1;BG1[2][23]=0;BG1[2][24]=0;BG1[2][25]=-1;
	BG1[3][22]=0;BG1[3][23]=1;BG1[3][24]=1;BG1[3][25]=0;
	for( int i=0;i<46;i++)
	{
		for( int j=0;j<68;j++)
		{
			I(BG1[i][j]%Z,Z);
			for( int i_=0;i_<Z;i_++)
				for( int j_=0;j_<Z;j_++)
					H[Z*i+i_][Z*j+j_]=mat[i_][j_];
		}
	}//end of outer-for.
	FILE* file=fopen("Parity.txt","w");
	for( int i=0;i<46*Z;i++)
	{
		for( int j=0;j<68*Z;j++)
			fprintf(file,"%d ",H[i][j]);
		fprintf(file,"\n");
	}
	fclose(file);
	VVI Gr(46*Z); //at max we will have 46*Z parity equations.
	int shazm=22;
	for( int i=0;i<46;i++)
	{
		for(int i_=0;i_<Z;i_++)
		{
			for( int j=0;j<shazm;j++)
			{
				for(int j_=0;j_<Z;j_++)
				{
					if(H[i*Z+i_][j*Z+j_]==1)
						Gr[i*Z+i_].push_back(j*Z+j_);
				}
			}//end of iterating through the columns.
		}
		shazm++;
	}//end of i,j.
	file=fopen("Generator.txt","w");
	print(file,Gr);
	fclose(file);
	return 0;
}
