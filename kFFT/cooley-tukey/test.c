// twiddle table, initialized by init_DFT(N)
double **DN;
void init_DFT(int N)
{ 
    int i, j, k, size_Dj = 16, n_max = log4(N);
    DN = malloc(sizeof(double*)*(n_max-1));
    for (j=1; j<n_max; j++, size_Dj*=4)
    { 
        double *Dj = DN[j-1] = malloc(2*sizeof(double)*size_Dj);
        for (k=0; k<size_Dj/4; k++){
            for (i=0; i<4; i++)
            { 
                *(Dj++) = cos(2*PI*i*k/size_Dj);
                *(Dj++) = sin(2*PI*i*k/size_Dj);
            }
        }
    }
}